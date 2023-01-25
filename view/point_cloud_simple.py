import argparse
import json
import os

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from dataset_tools.config import dataset_path, ycb_model_names, obj_ply_paths
from dataset_tools.loaders import load_intrinsics, get_depth_scale, get_camera_names, load_extrinsics, get_num_frame
from dataset_tools.view.open3d_window import Open3dWindow


class PointCloudWindow(Open3dWindow):
    def __init__(self, scene_name, init_frame_num, width=640*3+408, height=480*3):
        super().__init__(width, height, scene_name, 'Point Cloud')
        self.frame_num = init_frame_num
        self._update_frame_label()
        em = self.window.theme.font_size

        # view control
        view_ctrls = gui.CollapsableVert("View", 0, gui.Margins(em, 0, 0, 0))
        view_ctrls.set_is_open(True)
        self._settings_panel.add_child(view_ctrls)

        # pcd point size
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 5)
        self._point_size.set_on_value_changed(self._on_point_size)
        self._point_size.double_value = self.settings.scene_material.point_size
        grid = gui.VGrid(1, 0.25 * em)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        view_ctrls.add_child(grid)

        # select camera views
        view_ctrls.add_child(gui.Label("Active Cameras"))
        self.selected_cameras = []
        grid = gui.VGrid(2, 0.25 * em)
        view_ctrls.add_child(grid)
        for i in [0, 4, 1, 5, 2, 6, 3, 7]:
            box = gui.Checkbox(f'camera {i+1:02d}')
            box.checked = True
            box.set_on_checked(self._update_frame)
            grid.add_child(box)
            self.selected_cameras.append((i, box))

        self._update_frame()

    def _on_point_size(self, size):
        self.settings.scene_material.point_size = int(size)
        self.settings.apply_material = True
        self.scene_widget.scene.modify_geometry_material("pcd", self.settings.scene_material)

    def _set_camera_view(self):
        camera_name = get_camera_names(self.scene_path)[self.active_camera_view]
        intrinsic = load_intrinsics(f'{self.scene_path}/{camera_name}/camera_meta.yml')
        extrinsics = load_extrinsics(f'{self.scene_path}/cameras_meta.yml')
        extrinsic = np.linalg.inv(extrinsics[camera_name])
        self.scene_widget.setup_camera(intrinsic, extrinsic, 640, 480, self.bounds)

    def load_pcds(self, frame_num, active_camera_ids):
        pcds = o3d.geometry.PointCloud()
        for camera_name in np.array(get_camera_names(self.scene_path))[active_camera_ids]:
            intrinsic = load_intrinsics(f'{self.scene_path}/{camera_name}/camera_meta.yml')
            extrinsics = load_extrinsics(f'{self.scene_path}/cameras_meta.yml')
            extrinsic = np.linalg.inv(extrinsics[camera_name])
            depth_scale = get_depth_scale(f'{self.scene_path}/{camera_name}/camera_meta.yml', convert2unit='m')

            rgb_path = os.path.join(self.scene_path, f'{camera_name}', 'rgb', f'{frame_num:06}' + '.png')
            rgb_img = cv2.imread(rgb_path)
            depth_path = os.path.join(self.scene_path, f'{camera_name}', 'depth', f'{frame_num:06}' + '.png')
            depth_img = cv2.imread(depth_path, -1)
            depth_img = np.float32(depth_img * depth_scale)

            if self.mask_on:
                # read mask
                mask = xx
                apply_mask on rgb and depth

            pcds += load_pcd_from_rgbd(rgb_img, depth_img, intrinsic, extrinsic)
        return pcds

    def _update_frame(self, arg=None):
        self.scene_widget.scene.clear_geometry()
        self._update_frame_label()
        active_camera_ids = []
        for i, box in self.selected_cameras:
            if box.checked:
                active_camera_ids.append(i)
        pcd = self.load_pcds(self.frame_num, active_camera_ids)
        self.bounds = pcd.get_axis_aligned_bounding_box()
        self.scene_widget.scene.add_geometry("pcd", pcd, self.settings.scene_material)
        self._set_camera_view()

    def _on_next_frame(self):
        self.frame_num = min(get_num_frame(self.scene_path), self.frame_num + 1)
        self._update_frame()

    def _on_previous_frame(self):
        self.frame_num = max(0, self.frame_num - 1)
        self._update_frame()

    def _transform(self, event):
        if event.is_repeat:
            return gui.Widget.EventCallbackResult.HANDLED

        # Change camera view
        if event.key >= 49 and event.key <= 56:
            if event.type == gui.KeyEvent.DOWN:
                view_id = event.key - 49
                if view_id < len(self.camera_names):
                    print(f'Change to camera_{view_id + 1:02d}')
                    self.active_camera_view = view_id
                    self._update_frame()
                    return gui.Widget.EventCallbackResult.HANDLED
                else:
                    return gui.Widget.EventCallbackResult.HANDLED
            else:
                return gui.Widget.EventCallbackResult.HANDLED


def load_pcd_from_rgbd(rgb_img, depth_img, cam_K, extrinsic):
    rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    depth_img_o3d = o3d.geometry.Image(depth_img)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_img.shape[0], rgb_img.shape[1],
                                                  cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d,
                                                              depth_scale=1, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
    return pcd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_name', default='scene_2211192313')
    parser.add_argument('--start_frame', type=int, default=50)

    args = parser.parse_args()
    scene_name = args.scene_name
    start_image_num = args.start_frame

    gui.Application.instance.initialize()
    PointCloudWindow(scene_name, start_image_num)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
