import logging
import os
from copy import deepcopy

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from dataset_tools.config import dataset_path
from dataset_tools.utils.camera_parameter import load_extrinsics, get_depth_scale, load_intrinsics
from dataset_tools.utils.name import get_camera_names, get_num_frame, get_available_object_names
from dataset_tools.utils.pose import ObjectPoseTable


class Settings:
    def __init__(self):
        self.pcd_material = rendering.MaterialRecord()
        self.pcd_material.base_color = [0.9, 0.9, 0.9, 1.0]
        self.pcd_material.shader = "defaultUnlit"

        self.obj_material = rendering.MaterialRecord()
        # self.obj_material.base_color = [0, 1, 0, 1.0]
        self.obj_material.shader = "defaultUnlit"

        self.line_set_material = rendering.MaterialRecord()
        self.line_set_material.shader = "unlitLine"
        self.line_set_material.line_width = 1


class Open3dWindow:
    def __init__(self, width, height, scene_name, window_name, init_frame_num=0,
                 obj_pose_file=None, infra_pose_file=None):
        self.scene_name = scene_name
        self.scene_path = f'{dataset_path}/{scene_name}'
        self.active_camera_view = 0
        self.camera_names = get_camera_names(self.scene_path)
        self.frame_num = init_frame_num

        # load camera info
        self.intrinsics = {}
        self.extrinsics = load_extrinsics(f'{self.scene_path}/extrinsics.yml')
        self.depth_scale = get_depth_scale(f'{self.scene_path}/{self.camera_names[0]}/camera_meta.yml',
                                           convert2unit='m')
        for camera_name in self.camera_names:
            self.intrinsics[camera_name] = load_intrinsics(f'{self.scene_path}/{camera_name}/camera_meta.yml')
            self.extrinsics[camera_name] = np.linalg.inv(self.extrinsics[camera_name])
        # self.intrinsics = load_cameras_intrisics(scene_name)

        self.rgb_imgs = {}
        self.depth_imgs = {}

        self.settings = Settings()
        self.window = gui.Application.instance.create_window(window_name, width, height)

        em = self.window.theme.font_size

        # Settings panel
        self.settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.window.add_child(self.settings_panel)

        # Scene control
        self.scene_control = gui.CollapsableVert("Scene", 0.33 * em, gui.Margins(em, 0, 0, 0))
        self.scene_control.set_is_open(True)
        self.settings_panel.add_child(self.scene_control)

        # display scene name
        self.scene_label = gui.Label(self.scene_name)
        self.scene_control.add_child(self.scene_label)

        # display frame number
        self.frame_label = gui.Label('')
        self.update_frame_label()
        self.scene_control.add_child(self.frame_label)

        # frame navigation
        self.pre_frame_button = gui.Button("Previous frame")
        self.pre_frame_button.horizontal_padding_em = 0.8
        self.pre_frame_button.vertical_padding_em = 0
        self.pre_frame_button.set_on_clicked(self.on_previous_frame)
        self.next_frame_button = gui.Button("Next frame")
        self.next_frame_button.horizontal_padding_em = 0.8
        self.next_frame_button.vertical_padding_em = 0
        self.next_frame_button.set_on_clicked(self.on_next_frame)
        h = gui.Horiz(0.4 * em)
        h.add_stretch()
        h.add_child(self.pre_frame_button)
        h.add_child(self.next_frame_button)
        h.add_stretch()
        self.scene_control.add_child(h)
        self.scene_control.add_child(gui.VGrid(1, em))

        # View control
        self.view_ctrls = gui.CollapsableVert("View", 0, gui.Margins(em, 0, 0, 0))
        self.view_ctrls.set_is_open(True)
        self.settings_panel.add_child(self.view_ctrls)

        # pcd point size
        self.point_size = gui.Slider(gui.Slider.INT)
        self.point_size.set_limits(1, 5)
        self.point_size.set_on_value_changed(self.on_point_size)
        self.point_size.double_value = self.settings.pcd_material.point_size
        grid = gui.VGrid(1, 0.25 * em)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self.point_size)
        self.view_ctrls.add_child(grid)
        self.view_ctrls.add_child(gui.Label(""))

        # obj poses
        self.obj_pose_file = obj_pose_file
        self.obj_pose_box = gui.Checkbox(f'Objects')
        self.meshes = self.load_all_obj_meshes()
        self.view_ctrls.add_child(gui.Label(obj_pose_file))
        self.obj_pose_box.set_on_checked(self.update_frame)
        self.view_ctrls.add_child(self.obj_pose_box)
        self.view_ctrls.add_child(gui.Label(""))
        if obj_pose_file is not None:
            self.opt = ObjectPoseTable(f"{self.scene_path}/{self.camera_names[0]}/{obj_pose_file}",
                                              only_valid_pose=True)
            self.obj_pose_box.checked = True
        else:
            print('Using a temporary object pose table.')
            self.opt = ObjectPoseTable(scene_name=self.scene_name)
            self.obj_pose_box.checked = False

    def load_images(self):
        for camera_name in np.array(self.camera_names):
            rgb_path = os.path.join(self.scene_path, f'{camera_name}', 'rgb', f'{self.frame_num:06}' + '.png')
            self.rgb_imgs[camera_name] = cv2.imread(rgb_path)
            depth_path = os.path.join(self.scene_path, f'{camera_name}', 'depth', f'{self.frame_num:06}' + '.png')
            depth_img = cv2.imread(depth_path, -1)
            self.depth_imgs[camera_name] = np.float32(depth_img * self.depth_scale)

    def on_next_frame(self):
        self.frame_num = min(get_num_frame(self.scene_path) - 1, self.frame_num + 1)
        self.update_frame()

    def on_previous_frame(self):
        self.frame_num = max(0, self.frame_num - 1)
        self.update_frame()

    def update_frame_label(self):
        self.frame_label.text = f"Frame: {self.frame_num:06}"

    def on_point_size(self, size):
        pass

    def update_frame(self, frame=None):
        if frame is not None:
            self.frame_num = frame

    def load_all_obj_meshes(self):
        meshes = {}
        for object_name in get_available_object_names(self.scene_name):
            pcd_path = os.path.join(self.scene_path, 'models', object_name, 'object.pcd')
            if not os.path.exists(pcd_path):
                continue
            pcd = o3d.io.read_point_cloud(pcd_path)
            meshes[object_name] = pcd
        meshes['robot_tag'] = create_coordinate_frame_mesh(0.03)
        meshes['robot_base'] = create_coordinate_frame_mesh(0.2)
        meshes['sink_origin'] = create_coordinate_frame_mesh(0.06)
        return meshes

    def load_obj_mesh(self, object_name):
        if object_name not in self.meshes:
            logging.debug(f'no mesh for {object_name}')
            return None
        pose = self.opt.lookup(object_name, frame=self.frame_num)
        if len(pose) == 0:
            return None
        pose = pose[0]
        geometry = deepcopy(self.meshes[object_name])
        geometry.transform(pose)
        return geometry

    def on_keyboard_input(self, event):
        if event.is_repeat:
            return gui.Widget.EventCallbackResult.HANDLED

        # Change frame
        if event.key == gui.KeyName.LEFT:
            if event.type == gui.KeyEvent.DOWN:
                self.on_previous_frame()
            return True
        if event.key == gui.KeyName.RIGHT:
            if event.type == gui.KeyEvent.DOWN:
                self.on_next_frame()
            return True

        # Change camera view
        if 49 <= event.key <= 56:
            if event.type == gui.KeyEvent.DOWN:
                view_id = event.key - 49
                if view_id < len(self.camera_names):
                    self.on_change_active_camera_view(view_id)
                    return gui.Widget.EventCallbackResult.HANDLED
        return False

    def on_change_active_camera_view(self, camera_id):
        if camera_id < len(self.camera_names):
            print(f'Change to camera_{camera_id + 1:02d}')
            self.active_camera_view = camera_id
            self.set_active_camera_view()

    def set_active_camera_view(self):
        pass


def create_coordinate_frame_mesh(size=1.0):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    red_square = o3d.geometry.TriangleMesh.create_box(width=0.01, height=1.0, depth=1)
    red_square.translate([-0.005, -1.0 / 2, -1.0 / 2])
    red_square.paint_uniform_color([1, 0, 0])

    blue_square = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.01)
    blue_square.translate([-1.0 / 2, -1.0 / 2, -0.005])
    blue_square.paint_uniform_color([0, 0, 1])

    green_square = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.01, depth=1)
    green_square.translate([-1.0 / 2, -0.005, -1.0 / 2])
    green_square.paint_uniform_color([0, 1, 0])

    mesh = coordinate_frame + red_square + blue_square + green_square
    mesh.scale(size, [0, 0, 0])
    return mesh


if __name__ == "__main__":
    scene_name = 'scene_230905145629'
    gui.Application.instance.initialize()
    w = Open3dWindow(2048, 1536, scene_name, 'test')
    gui.Application.instance.run()
