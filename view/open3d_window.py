import argparse
import os
from copy import deepcopy

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from dataset_tools.config import dataset_path, obj_ply_paths
from dataset_tools.loaders import get_camera_names, load_intrinsics, load_extrinsics, get_depth_scale, \
    load_object_pose_table, get_num_frame


class Settings:
    def __init__(self):
        self.pcd_material = rendering.MaterialRecord()
        self.pcd_material.base_color = [0.9, 0.9, 0.9, 1.0]
        self.pcd_material.shader = "defaultUnlit"

        self.obj_material = rendering.MaterialRecord()
        self.obj_material.base_color = [0, 1, 0, 1.0]
        self.obj_material.shader = "defaultUnlit"

        self.line_set_material = rendering.MaterialRecord()
        self.line_set_material.shader = "unlitLine"
        self.line_set_material.line_width = 1


class Open3dWindow:
    def __init__(self, width, height, scene_name, window_name, init_frame_num=0, obj_pose_file=None):
        self.scene_name = scene_name
        self.scene_path = f'{dataset_path}/{scene_name}'
        self.active_camera_view = 0
        self.camera_names = get_camera_names(self.scene_path)
        self.frame_num = init_frame_num

        # load camera info
        self.intrinsics = {}
        self.extrinsics = load_extrinsics(f'{self.scene_path}/extrinsics.yml')
        self.depth_scale = get_depth_scale(f'{self.scene_path}/{self.camera_names[0]}/camera_meta.yml', convert2unit='m')
        for camera_name in self.camera_names:
            self.intrinsics[camera_name] = load_intrinsics(f'{self.scene_path}/{camera_name}/camera_meta.yml')
            self.extrinsics[camera_name] = np.linalg.inv(self.extrinsics[camera_name])

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
        if obj_pose_file is not None:
            self.opt = load_object_pose_table(f"{self.scene_path}/{self.camera_names[0]}/{obj_pose_file}",
                                              only_valid_pose=True)
            self.meshes = self.load_all_obj_meshes()
            self.view_ctrls.add_child(gui.Label(obj_pose_file))
            self.obj_pose_box.set_on_checked(self.update_frame)
            self.view_ctrls.add_child(self.obj_pose_box)
            self.view_ctrls.add_child(gui.Label(""))
            self.obj_id_meshes = []

    def load_images(self):
        for camera_name in np.array(self.camera_names):
            rgb_path = os.path.join(self.scene_path, f'{camera_name}', 'rgb', f'{self.frame_num:06}' + '.png')
            self.rgb_imgs[camera_name] = cv2.imread(rgb_path)
            depth_path = os.path.join(self.scene_path, f'{camera_name}', 'depth', f'{self.frame_num:06}' + '.png')
            depth_img = cv2.imread(depth_path, -1)
            self.depth_imgs[camera_name] = np.float32(depth_img * self.depth_scale)

    def on_next_frame(self):
        self.frame_num = min(get_num_frame(self.scene_path), self.frame_num + 1)
        self.update_frame()

    def on_previous_frame(self):
        self.frame_num = max(0, self.frame_num - 1)
        self.update_frame()

    def update_frame_label(self):
        self.frame_label.text = f"Frame: {self.frame_num:06}"

    def on_point_size(self, size):
        pass

    def update_frame(self, args=None):
        pass

    def load_all_obj_meshes(self):
        meshes = {}
        for id, p in obj_ply_paths.items():
            geometry = o3d.io.read_point_cloud(p)
            geometry.points = o3d.utility.Vector3dVector(np.array(geometry.points) / 1000)
            meshes[id] = geometry
        return meshes

    def load_obj_mesh(self, obj_id):
        d = np.logical_and(self.opt['frame'] == self.frame_num, self.opt['obj_id'] == obj_id)
        t = self.opt[d]
        if len(t) == 0:
            return None
        assert len(t) == 1
        t = t[0]

        geometry = deepcopy(self.meshes[obj_id])
        pose = t['pose']
        geometry.translate(pose[0:3, 3] / 1000)
        center = geometry.get_center()
        geometry.rotate(pose[0:3, 0:3], center=center)
        return geometry

    def on_keyboard_input(self, event):
        pass

    def on_change_active_camera_view(self, camera_id):
        if camera_id < len(self.camera_names):
            print(f'Change to camera_{camera_id + 1:02d}')
            self.active_camera_view = camera_id
            self.set_active_camera_view()

    def set_active_camera_view(self):
        pass


if __name__ == "__main__":
    scene_name = 'scene_2211192313'
    gui.Application.instance.initialize()
    w = Open3dWindow(2048, 1536, scene_name, 'test')
    gui.Application.instance.run()
