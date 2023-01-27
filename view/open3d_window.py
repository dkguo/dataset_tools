import argparse
import os

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from dataset_tools.config import dataset_path
from dataset_tools.loaders import get_camera_names, load_intrinsics, load_extrinsics, get_depth_scale


class Settings:
    UNLIT = "defaultUnlit"

    def __init__(self):
        self.bg_color = gui.Color(1, 1, 1)
        self.show_axes = False
        self.highlight_obj = True

        self.apply_material = True  # clear to False after processing

        self.scene_material = rendering.MaterialRecord()
        self.scene_material.base_color = [0.9, 0.9, 0.9, 1.0]
        self.scene_material.shader = Settings.UNLIT

        self.annotation_obj_material = rendering.MaterialRecord()
        self.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
        self.annotation_obj_material.shader = Settings.UNLIT


class Open3dWindow:
    def __init__(self, width, height, scene_name, window_name):
        self.scene_name = scene_name
        self.scene_path = f'{dataset_path}/{scene_name}'
        self.active_camera_view = 0
        self.camera_names = get_camera_names(self.scene_path)
        self.frame_num = 0

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
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)

        w = self.window
        em = w.theme.font_size

        # Settings panel
        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        w.set_on_layout(self._on_layout)
        w.add_child(self._settings_panel)

        # Scene control
        self._scene_control = gui.CollapsableVert("Scene", 0.33 * em, gui.Margins(em, 0, 0, 0))
        self._scene_control.set_is_open(True)
        self._settings_panel.add_child(self._scene_control)

        # display scene name
        self._scene_label = gui.Label(self.scene_name)
        self._scene_control.add_child(self._scene_label)

        # display frame number
        self._frame_label = gui.Label('')
        self._update_frame_label()
        self._scene_control.add_child(self._frame_label)

        # frame navigation
        self._pre_frame_button = gui.Button("Previous frame")
        self._pre_frame_button.horizontal_padding_em = 0.8
        self._pre_frame_button.vertical_padding_em = 0
        self._pre_frame_button.set_on_clicked(self._on_previous_frame)
        self._next_frame_button = gui.Button("Next frame")
        self._next_frame_button.horizontal_padding_em = 0.8
        self._next_frame_button.vertical_padding_em = 0
        self._next_frame_button.set_on_clicked(self._on_next_frame)
        h = gui.Horiz(0.4 * em)
        h.add_stretch()
        h.add_child(self._pre_frame_button)
        h.add_child(self._next_frame_button)
        h.add_stretch()
        self._scene_control.add_child(h)
        self.scene_widget.set_on_key(self._transform)

    def load_images(self):
        for camera_name in np.array(self.camera_names):
            rgb_path = os.path.join(self.scene_path, f'{camera_name}', 'rgb', f'{self.frame_num:06}' + '.png')
            self.rgb_imgs[camera_name] = cv2.imread(rgb_path)
            depth_path = os.path.join(self.scene_path, f'{camera_name}', 'depth', f'{self.frame_num:06}' + '.png')
            depth_img = cv2.imread(depth_path, -1)
            self.depth_imgs[camera_name] = np.float32(depth_img * self.depth_scale)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        width = 17 * layout_context.theme.font_size
        height = min(r.height,
                     self._settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)
        self.scene_widget.frame = gui.Rect(0, r.y, r.get_right() - width, r.height)

    def _on_previous_frame(self):
        pass

    def _on_next_frame(self):
        pass

    def _update_frame_label(self):
        self._frame_label.text = f"Frame: {self.frame_num:06}"

    def _transform(self, event):
        pass


if __name__ == "__main__":
    scene_name = 'scene_2211192313'
    gui.Application.instance.initialize()
    w = Open3dWindow(2048, 1536, scene_name, 'test')
    gui.Application.instance.run()
