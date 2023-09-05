# from dataset_tools.view.renderer import create_renderer
# create_renderer()
from copy import deepcopy

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

from dataset_tools.config import ycb_model_names, obj_model_names
from dataset_tools.utils import save_object_pose_table
from dataset_tools.view.open3d_window import Open3dWindow
from dataset_tools.view.point_cloud import load_pcd_from_rgbd


class Annotation(Open3dWindow):
    def __init__(self, scene_name, init_frame_num, width=640 * 3 + 408, height=480 * 3, init_obj_pose_file=None):
        super().__init__(width, height, scene_name, 'Point Cloud', init_frame_num, obj_pose_file=init_obj_pose_file)
        em = self.window.theme.font_size

        self.window.set_on_key(self.on_keyboard_input)

        self.dist = 0.005
        self.deg = 1.

        self.left_shift_modifier = False
        self.annotation_changed = False

        self.objects = gui.CollapsableVert("Objects", 0.33 * em, gui.Margins(em, 0, 0, 0))
        self.meshes_available = gui.ListView()
        self.meshes_available.set_max_visible_items(10)
        self.meshes_available.set_items(ycb_model_names)
        self.meshes_used = gui.ListView()
        self.meshes_used.set_max_visible_items(5)
        self.add_mesh_button = gui.Button("Add Mesh")
        self.add_mesh_button.set_on_clicked(self.on_add_mesh)
        self.remove_mesh_button = gui.Button("Remove Mesh")
        self.remove_mesh_button.set_on_clicked(self.on_remove_mesh)
        self.save_obj_pose_button = gui.Button("Save Object Pose")
        self.save_obj_pose_button.set_on_clicked(self.on_save_obj_pose)
        self.save_infra_pose_button = gui.Button("Save Infrastructure Pose")
        self.save_infra_pose_button.set_on_clicked(self.on_save_infra_pose)
        self.objects.add_child(self.meshes_available)
        self.objects.add_child(self.add_mesh_button)
        self.objects.add_child(self.meshes_used)
        self.objects.add_child(self.remove_mesh_button)
        self.objects.add_child(self.save_obj_pose_button)
        self.objects.add_child(self.save_infra_pose_button)
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
            col = i // 3
            row = i % 3
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
                self.objects.set_is_open(True)
                obj_ids = self.opt[self.opt['frame'] == self.frame_num]['obj_id']
                for obj_id in obj_ids:
                    mesh = self.load_obj_mesh(obj_id)
                    self.scene_widgets[i].scene.add_geometry(str(obj_id), mesh, self.settings.obj_material)
                self.meshes_used.set_items([f'{i:03d}_{obj_model_names[i]}' for i in obj_ids])
                self.meshes_used.selected_index = 0
            else:
                self.meshes_used.set_items([])
                self.objects.set_is_open(False)

    def on_add_mesh(self):
        if self.meshes_available.selected_index == -1:
            print('no obj selected')
            return

        obj_id = int(self.meshes_available.selected_value[:3])

        if obj_id in self.opt[self.opt['frame'] == self.frame_num]['obj_id']:
            print(f'{obj_model_names[obj_id]} already exists')
            return

        if self.opt.size == 1 and self.opt[0]['obj_id'] == -1:
            pass
        else:
            self.opt = np.append(self.opt, self.opt[-1])
        self.opt[-1]['frame'] = self.frame_num
        self.opt[-1]['obj_id'] = obj_id

        print(self.opt)
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

    def on_save_obj_pose(self):
        opt = self.opt[self.opt['obj_id'] < 99]
        save_object_pose_table(opt, f'{self.scene_path}/object_pose/ground_truth_tmp.csv')
        self.annotation_changed = False
        print('object pose saved')

    def on_save_infra_pose(self):
        ipt = self.opt[self.opt['obj_id'] > 99]
        save_object_pose_table(ipt, f'{self.scene_path}/infra_poses_tmp.csv')
        self.annotation_changed = False
        print('infrastructure pose saved')

    def on_keyboard_input(self, event):
        if super().on_keyboard_input(event):
            return True

        # if ctrl is pressed then increase translation and angle values
        if event.key == gui.KeyName.LEFT_CONTROL:
            if event.type == gui.KeyEvent.DOWN:
                self.dist = 0.02
                self.deg = 5
            elif event.type == gui.KeyEvent.UP:
                self.dist = 0.05
                self.deg = 30
            return True

        if event.key == gui.KeyName.ALT:
            if event.type == gui.KeyEvent.DOWN:
                self.dist = 0.05
                self.deg = 30
            elif event.type == gui.KeyEvent.UP:
                self.dist = 0.005
                self.deg = 1.
            return True

        # if no active_mesh selected print error
        if self.meshes_used.selected_index == -1:
            print("No objects are selected in scene meshes")
            return True

        if event.type == gui.KeyEvent.DOWN:
            # copy from previous/next frame
            if event.key == gui.KeyName.B:
                print("B pressed: copy from previous frame")
                self.copy_pose_from_frame(self.frame_num - 1)
            elif event.key == gui.KeyName.N:
                print("N pressed: copy from next frame")
                self.copy_pose_from_frame(self.frame_num + 1)

            # Translate object
            if not self.left_shift_modifier:
                if event.key == gui.KeyName.J:
                    print("J pressed: translate to left")
                    self.move_selected_obj(-self.dist, 0, 0, 0, 0, 0)
                elif event.key == gui.KeyName.L:
                    print("L pressed: translate to right")
                    self.move_selected_obj(self.dist, 0, 0, 0, 0, 0)
                elif event.key == gui.KeyName.K:
                    print("K pressed: translate down")
                    self.move_selected_obj(0, self.dist, 0, 0, 0, 0)
                elif event.key == gui.KeyName.I:
                    print("I pressed: translate up")
                    self.move_selected_obj(0, -self.dist, 0, 0, 0, 0)
                elif event.key == gui.KeyName.U:
                    print("U pressed: translate furthur")
                    self.move_selected_obj(0, 0, self.dist, 0, 0, 0)
                elif event.key == gui.KeyName.O:
                    print("O pressed: translate closer")
                    self.move_selected_obj(0, 0, -self.dist, 0, 0, 0)

            # Rotation - keystrokes are not in same order as translation to make movement more human intuitive
            else:
                print("Left-Shift is clicked; rotation mode")
                if event.key == gui.KeyName.O:
                    print("O pressed: rotate CW")
                    self.move_selected_obj(0, 0, 0, 0, 0, self.deg * np.pi / 180)
                elif event.key == gui.KeyName.U:
                    print("U pressed: rotate CCW")
                    self.move_selected_obj(0, 0, 0, 0, 0, -self.deg * np.pi / 180)
                elif event.key == gui.KeyName.J:
                    print("J pressed: rotate towards left")
                    self.move_selected_obj(0, 0, 0, 0, self.deg * np.pi / 180, 0)
                elif event.key == gui.KeyName.L:
                    print("L pressed: rotate towards right")
                    self.move_selected_obj(0, 0, 0, 0, -self.deg * np.pi / 180, 0)
                elif event.key == gui.KeyName.K:
                    print("K pressed: rotate downwards")
                    self.move_selected_obj(0, 0, 0, self.deg * np.pi / 180, 0, 0)
                elif event.key == gui.KeyName.I:
                    print("I pressed: rotate upwards")
                    self.move_selected_obj(0, 0, 0, -self.deg * np.pi / 180, 0, 0)

        return True

    def move_selected_obj(self, x, y, z, rx, ry, rz):
        self.annotation_changed = True

        obj_id = int(self.meshes_used.selected_value[:3])
        d = np.logical_and(self.opt['frame'] == self.frame_num, self.opt['obj_id'] == obj_id)
        assert len(d.nonzero()[0] == 1)
        d = d.nonzero()[0][0]
        pose = self.opt[d]['pose']

        geometry = deepcopy(self.meshes[obj_id])
        geometry.transform(pose)

        T_ci_to_c0 = self.extrinsics[self.camera_names[self.active_camera_view]]

        # translation or rotation
        if x != 0 or y != 0 or z != 0:
            ci_h_transform = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        else:
            c0_center = geometry.get_center()
            ci_center = T_ci_to_c0 @ np.append(c0_center, 1)
            ci_center = ci_center[:3]

            ci_T_neg = np.vstack((np.hstack((np.identity(3), -ci_center.reshape(3, 1))), [0, 0, 0, 1]))
            ci_T_pos = np.vstack((np.hstack((np.identity(3), ci_center.reshape(3, 1))), [0, 0, 0, 1]))

            rot_mat_obj_center = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
            R = np.vstack((np.hstack((rot_mat_obj_center, [[0], [0], [0]])), [0, 0, 0, 1]))
            ci_h_transform = np.matmul(ci_T_pos, np.matmul(R, ci_T_neg))

        h_transform = np.linalg.inv(T_ci_to_c0) @ ci_h_transform @ T_ci_to_c0

        geometry.transform(h_transform)
        new_pose = h_transform @ pose
        self.opt[d]['pose'] = new_pose

        # update every scene widget
        for w in self.scene_widgets:
            w.scene.remove_geometry(str(obj_id))
            w.scene.add_geometry(str(obj_id), geometry, self.settings.obj_material)

    def copy_pose_from_frame(self, src_frame_num):
        self.annotation_changed = True

        obj_id = int(self.meshes_used.selected_value[:3])

        src_row = np.logical_and(self.opt['frame'] == src_frame_num, self.opt['obj_id'] == obj_id)
        if not len(src_row.nonzero()[0] == 1):
            print(f"Object {obj_id} not found in frame {src_frame_num}")
            return
        src_row = src_row.nonzero()[0][0]
        queued_pose = self.opt[src_row]['pose']

        dest_row = np.logical_and(self.opt['frame'] == self.frame_num, self.opt['obj_id'] == obj_id).nonzero()[0][0]
        self.opt[dest_row]['pose'] = deepcopy(queued_pose)

        self.update_frame()


if __name__ == "__main__":
    scene_name = 'scene_230704142825'
    start_image_num = 370
    hand_mask_dir = 'hand_pose/d2/mask'
    # init_obj_pose_file = 'object_pose/multiview_medium/object_poses.csv'
    init_obj_pose_file = '../object_pose/ground_truth.csv'
    # init_obj_pose_file = None

    gui.Application.instance.initialize()
    w = Annotation(scene_name, start_image_num, init_obj_pose_file=init_obj_pose_file)

    gui.Application.instance.run()
