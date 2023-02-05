# from dataset_tools.view.renderer import create_renderer
# create_renderer()
import numpy as np
import open3d as o3d

from open3d.visualization import gui, rendering

from dataset_tools.config import ycb_model_names, obj_model_names
from dataset_tools.loaders import load_intrinsics, load_extrinsics
from dataset_tools.view.open3d_window import Open3dWindow
from dataset_tools.view.point_cloud_simple import load_pcd_from_rgbd, PointCloudWindow


class Annotation(Open3dWindow):
    def __init__(self, scene_name, init_frame_num, width=640*3+408, height=480*3, hand_mask_dir=None, obj_pose_file=None):
        super().__init__(width, height, scene_name, 'Point Cloud', init_frame_num, obj_pose_file=obj_pose_file)
        em = self.window.theme.font_size

        self.obj_pose_box.checked = True

        self.objects = gui.CollapsableVert("Objects", 0.33 * em, gui.Margins(em, 0, 0, 0))
        self.objects.set_is_open(True)
        self.meshes_available = gui.ListView()
        self.meshes_available.set_max_visible_items(10)
        self.meshes_available.set_items(ycb_model_names)
        self.meshes_used = gui.ListView()
        self.meshes_used.set_max_visible_items(5)
        self.add_mesh_button = gui.Button("Add Mesh")
        self.remove_mesh_button = gui.Button("Remove Mesh")
        self.add_mesh_button.set_on_clicked(self.on_add_mesh)
        self.remove_mesh_button.set_on_clicked(self.on_remove_mesh)
        self.objects.add_child(self.meshes_available)
        self.objects.add_child(self.add_mesh_button)
        self.objects.add_child(self.meshes_used)
        self.objects.add_child(self.remove_mesh_button)
        self.settings_panel.add_child(self.objects)

        self.scene_widgets = []
        for camera_name in self.camera_names:
            w = gui.SceneWidget()
            w.scene = rendering.Open3DScene(self.window.renderer)
            self.window.add_child(w)
            self.scene_widgets.append(w)

        self.window.set_on_layout(self.on_layout)

        self.frame_num = init_frame_num
        self.update_frame()

    def on_layout(self, layout_context):
        r = self.window.content_rect
        width = 17 * layout_context.theme.font_size
        height = max(r.height,
                     self.settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        self.settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)
        w = (r.get_right() - width) // 3
        h = r.height // 3
        assert len(self.scene_widgets) == 8
        for i, s in enumerate(self.scene_widgets):
            row = i // 3
            col = i % 3
            s.frame = gui.Rect(row * w, col * h, w, h)

    def on_point_size(self, size):
        self.settings.pcd_material.point_size = int(size)
        for s in self.scene_widgets:
            s.scene.modify_geometry_material("pcd", self.settings.pcd_material)

    def load_pcds(self):
        pcds = []
        for camera_name in self.camera_names:
            intrinsic = self.intrinsics[camera_name]
            extrinsic = self.extrinsics[camera_name]
            rgb_img = self.rgb_imgs[camera_name]
            depth_img = self.depth_imgs[camera_name]

            pcds.append(load_pcd_from_rgbd(rgb_img, depth_img, intrinsic, extrinsic))
        return pcds

    def update_frame(self, args=None):
        self.update_frame_label()
        self.load_images()

        pcds = self.load_pcds()

        for i, camera_name in enumerate(self.camera_names):
            self.scene_widgets[i].scene.clear_geometry()
            self.scene_widgets[i].scene.add_geometry("pcd", pcds[i], self.settings.pcd_material)
            intrinsic = self.intrinsics[camera_name]
            extrinsic = self.extrinsics[camera_name]
            bounds = pcds[i].get_axis_aligned_bounding_box()
            self.scene_widgets[i].setup_camera(intrinsic, extrinsic, 640, 480, bounds)

            if self.obj_pose_box.checked:
                obj_ids = self.opt[self.opt['frame'] == self.frame_num]['obj_id']
                for obj_id in obj_ids:
                    mesh = self.load_obj_mesh(obj_id)
                    self.scene_widgets[i].scene.add_geometry(str(obj_id), mesh, self.settings.obj_material)
                self.meshes_used.set_items([obj_model_names[i] for i in obj_ids])
            else:
                self.meshes_used.set_items([])

    def on_add_mesh(self):
        if self.meshes_available.selected_index == -1:
            print('no obj selected')
            return

        obj_id = int(self.meshes_available.selected_value[:3])

        if obj_id in self.opt[self.opt['frame'] == self.frame_num]['obj_id']:
            print(f'{obj_model_names[obj_id]} already exists')
            return

        self.opt = np.append(self.opt, self.opt[-1])
        self.opt[-1]['frame'] = self.frame_num
        self.opt[-1]['obj_id'] = obj_id

        print(f'{obj_model_names[obj_id]} added')
        self.update_frame()

    def on_remove_mesh(self):
        if self.meshes_used.selected_index == -1:
            print('no obj selected')
            return
        obj_id = int(self.meshes_used.selected_value[:3])
        s = np.logical_and(self.opt['frame'] == self.frame_num, self.opt['obj_id'] == obj_id)
        self.opt = self.opt[~s]
        print(f'{obj_model_names[obj_id]} removed')
        self.update_frame()



if __name__ == "__main__":
    scene_name = 'scene_2210232307_01'
    start_image_num = 40
    hand_mask_dir = 'hand_pose/d2/mask'
    obj_pose_file = 'object_pose/multiview_medium/object_poses.csv'


    gui.Application.instance.initialize()
    w = Annotation(scene_name, start_image_num, obj_pose_file=obj_pose_file)

    gui.Application.instance.run()