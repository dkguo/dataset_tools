import argparse
import json
import os

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from dataset_tools.config import dataset_path, ycb_model_names, obj_ply_paths
from dataset_tools.loaders import load_intrinsics, get_depth_scale, get_camera_names, load_extrinsics
from dataset_tools.view.open3d_window import Open3dWindow


class PointCloudWindow(Open3dWindow):
    def __init__(self, scene_name, init_frame_num, width=2000, height=1400):
        super().__init__(width, height, scene_name, 'Point Cloud')
        self.frame_num = init_frame_num
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
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        view_ctrls.add_child(grid)

        self.load_pcds(self.frame_num)

    def _on_point_size(self, size):
        self.settings.scene_material.point_size = int(size)
        self.settings.apply_material = True
        self.scene_widget.scene.modify_geometry_material("annotation_scene", self.settings.scene_material)

    def set_camera(self, pcd, intrinsic, extrinsic):
        bounds = pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(intrinsic, extrinsic, 640, 480, bounds)

    def load_pcds(self, frame_num):
        pcds = o3d.geometry.PointCloud()
        for camera_name in get_camera_names(self.scene_path):
            intrinsic = load_intrinsics(f'{self.scene_path}/{camera_name}/camera_meta.yml')
            extrinsics = load_extrinsics(f'{self.scene_path}/cameras_meta.yml')
            extrinsic = np.linalg.inv(extrinsics[camera_name])
            depth_scale = get_depth_scale(f'{self.scene_path}/{camera_name}/camera_meta.yml', convert2unit='m')

            rgb_path = os.path.join(self.scene_path, f'{camera_name}', 'rgb', f'{frame_num:06}' + '.png')
            rgb_img = cv2.imread(rgb_path)
            depth_path = os.path.join(self.scene_path, f'{camera_name}', 'depth', f'{frame_num:06}' + '.png')
            depth_img = cv2.imread(depth_path, -1)
            depth_img = np.float32(depth_img * depth_scale)

            pcds += load_pcd_from_rgbd(rgb_img, depth_img, intrinsic, extrinsic)
        self.scene_widget.scene.add_geometry("pcd", pcds, self.settings.scene_material)
        self.set_camera(pcds, intrinsic, extrinsic)


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
