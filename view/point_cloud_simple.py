import argparse
import json
import os

import cv2
import numpy as np
# from dataset_tools.view.open3ddev.geometry import BoundingConvexHull
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import pandas as pd
from tqdm import tqdm

from dataset_tools.config import dataset_path, ycb_model_names, obj_ply_paths, resolution_width, resolution_height
from dataset_tools.loaders import load_intrinsics, get_depth_scale, get_camera_names, load_extrinsics, get_num_frame, \
    load_object_pose_table
from dataset_tools.view.open3d_window import Open3dWindow


class PointCloudWindow(Open3dWindow):
    def __init__(self, scene_name, init_frame_num, width=640*3+408, height=480*3, hand_mask_dir=None, obj_pose_file=None):
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
        self._point_size.double_value = self.settings.pcd_material.point_size
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

        # mask
        self.hand_mask_dir = hand_mask_dir
        self.hand_mask_box = gui.Checkbox(f'Apply Masks')
        if hand_mask_dir is not None:
            self.hand_masks = {}
            view_ctrls.add_child(gui.Label(hand_mask_dir))
            self.hand_mask_box.set_on_checked(self._update_frame)
            view_ctrls.add_child(self.hand_mask_box)

            # intersection
            self.hand_mask_intsct_box = gui.Checkbox(f'Intersect Masks')
            self.hand_mask_intsct_box.set_on_checked(self._update_frame)
            view_ctrls.add_child(self.hand_mask_intsct_box)
            self.hand_mask_intsct_convex_hull_box = gui.Checkbox(f'Show convex hull')
            self.hand_mask_intsct_convex_hull_box.set_on_checked(self._update_frame)
            view_ctrls.add_child(self.hand_mask_intsct_convex_hull_box)
        
        # obj poses
        self.obj_pose_file = obj_pose_file
        self.obj_pose_box = gui.Checkbox(f'Objects')
        if obj_pose_file is not None:
            self.opt = load_object_pose_table(f"{self.scene_path}/{self.camera_names[0]}/{obj_pose_file}",
                                              only_valid_pose=True)
            view_ctrls.add_child(gui.Label(obj_pose_file))
            self.obj_pose_box.set_on_checked(self._update_frame)
            view_ctrls.add_child(self.obj_pose_box)

        self._update_frame()
        self._set_camera_view()

    def _on_point_size(self, size):
        self.settings.pcd_material.point_size = int(size)
        self.settings.apply_material = True
        self.scene_widget.scene.modify_geometry_material("pcd", self.settings.pcd_material)

    def _set_camera_view(self):
        camera_name = get_camera_names(self.scene_path)[self.active_camera_view]
        intrinsic = load_intrinsics(f'{self.scene_path}/{camera_name}/camera_meta.yml')
        extrinsics = load_extrinsics(f'{self.scene_path}/extrinsics.yml')
        extrinsic = np.linalg.inv(extrinsics[camera_name])
        self.scene_widget.setup_camera(intrinsic, extrinsic, 640, 480, self.bounds)

    def load_pcds(self):
        pcds = o3d.geometry.PointCloud()
        for camera_name in self._get_selected_camera_names():
            intrinsic = self.intrinsics[camera_name]
            extrinsic = self.extrinsics[camera_name]
            rgb_img = self.rgb_imgs[camera_name]
            depth_img = self.depth_imgs[camera_name]

            if self.hand_mask_box.checked:
                mask = self.hand_masks[camera_name]
                rgb_img = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
                depth_img = cv2.bitwise_and(depth_img, depth_img, mask=mask)

            pcds += load_pcd_from_rgbd(rgb_img, depth_img, intrinsic, extrinsic)
        return pcds

    def load_hand_masks(self):
        for camera_name in self.camera_names:
            mask_path = f'{self.scene_path}/{camera_name}/{self.hand_mask_dir}/{self.frame_num:06}.png'
            if os.path.exists(mask_path):
                self.hand_masks[camera_name] = cv2.imread(mask_path, 0)
            else:
                self.hand_masks[camera_name] = np.zeros((resolution_height, resolution_width)).astype('uint8')
    
    def load_obj_meshes(self):
        obj_id_meshes = []
        for t in self.opt[self.opt['frame'] == self.frame_num]:
            obj_id = t['obj_id']
            if obj_id == 26:
                continue
            geometry = o3d.io.read_point_cloud(obj_ply_paths[obj_id])
            pose = t['pose']
            geometry.translate(pose[0:3, 3])
            center = geometry.get_center()
            geometry.rotate(pose[0:3, 0:3], center=center)
            geometry.points = o3d.utility.Vector3dVector(np.array(geometry.points) / 1000)
            obj_id_meshes.append((obj_id, geometry))
        return obj_id_meshes

    def load_convex_hulls_ls(self):
        hulls = []
        hull_lss = []
        for camera_name in self._get_selected_camera_names():
            intrinsic = self.intrinsics[camera_name]
            extrinsic = self.extrinsics[camera_name]
            mask = self.hand_masks[camera_name]
            if mask.max() == 0:
                continue
            hull, hull_ls = compute_convex_hull_line_set_from_mask(mask, intrinsic, extrinsic)
            if hull is not None:
                hulls.append(hull)
                hull_lss.append(hull_ls)
        return hulls, hull_lss

    def _update_frame(self, arg=None):
        self.scene_widget.scene.clear_geometry()
        self._update_frame_label()
        self.load_images()
        self.load_hand_masks()
        pcd = self.load_pcds()

        if self.hand_mask_intsct_box.checked:
            convex_hulls, hull_lss = self.load_convex_hulls_ls()
            pcd = crop_pcd_by_convex_hulls(pcd, convex_hulls)
            if self.hand_mask_intsct_convex_hull_box.checked:
                for i, hull_ls in enumerate(hull_lss):
                    self.scene_widget.scene.add_geometry(f"hull_{i}", hull_ls, self.settings.line_set_material)

        self.bounds = pcd.get_axis_aligned_bounding_box()
        self.scene_widget.scene.add_geometry("pcd", pcd, self.settings.pcd_material)

        # add objs
        if self.obj_pose_box.checked:
            self.obj_id_meshes = self.load_obj_meshes()
            for obj_id, mesh in self.obj_id_meshes:
                self.scene_widget.scene.add_geometry(str(obj_id), mesh, self.settings.obj_material)

        # calculate distance
        if self.hand_mask_intsct_box.checked and self.obj_pose_box.checked and len(convex_hulls) > 0:
            calculate_hand_obj_distance(pcd, self.obj_id_meshes)

    def _on_next_frame(self):
        self.frame_num = min(get_num_frame(self.scene_path), self.frame_num + 1)
        self._update_frame()

    def _on_previous_frame(self):
        self.frame_num = max(0, self.frame_num - 1)
        self._update_frame()

    def _on_keyboard_input(self, event):
        if event.is_repeat:
            return gui.Widget.EventCallbackResult.HANDLED

        # Change camera view
        if event.key >= 49 and event.key <= 56:
            if event.type == gui.KeyEvent.DOWN:
                view_id = event.key - 49
                if view_id < len(self.camera_names):
                    print(f'Change to camera_{view_id + 1:02d}')
                    self.active_camera_view = view_id
                    self._set_camera_view()
                    return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.HANDLED

    def _get_selected_camera_names(self):
        active_camera_ids = []
        for i, box in self.selected_cameras:
            if box.checked:
                active_camera_ids.append(i)
        return np.array(self.camera_names)[active_camera_ids]

    def save_hand_obj_dists(self, save_file_path):
        detections = np.empty(0, dtype=[('scene_name', 'U20'),
                                        ('camera_name', 'U22'),
                                        ('frame', 'i4'),
                                        ('obj_id', 'i2'),
                                        ('in_contact', 'i1'),
                                        ('distance', 'f')])

        for frame in tqdm(range(get_num_frame(self.scene_path))):
            self.frame_num = frame
            self.load_images()
            self.load_hand_masks()
            pcd = self.load_pcds()
            convex_hulls, hull_lss = self.load_convex_hulls_ls()
            pcd = crop_pcd_by_convex_hulls(pcd, convex_hulls)
            obj_id_meshes = self.load_obj_meshes()
            if len(convex_hulls) > 0:
                obj_id_dists = calculate_hand_obj_distance(pcd, obj_id_meshes)
                for obj_id, dist in obj_id_dists:
                    if dist < 0.02:
                        detections = np.append(detections,
                                               np.array([(self.scene_name, 'combined', frame, obj_id, 1, dist)],
                                                        dtype=detections.dtype))
                    else:
                        detections = np.append(detections,
                                               np.array([(self.scene_name, 'combined', frame, obj_id, 0, dist)],
                                                        dtype=detections.dtype))

        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        df = pd.DataFrame.from_records(detections)
        df.to_csv(save_file_path, index=False)


def calculate_hand_obj_distance(hand_pcd, obj_id_meshes):
    id_dist = []
    for obj_id, mesh in obj_id_meshes:
        dist = min(hand_pcd.compute_point_cloud_distance(mesh))
        id_dist.append((obj_id, dist))
        # print(obj_id, dist)
    return id_dist


def crop_pcd_by_convex_hulls(pcd, convex_hulls):
    for hull in convex_hulls:
        pcd = pcd.crop_convex_hull(hull)
    return pcd


def compute_convex_hull_line_set_from_mask(mask, intrinsic, extrinsic, min_d=0.1, max_d=2.0,
                                           boundrary_thres=5, area_thres=0.25):
    if np.sum(mask > 0) / mask.size > 0.25:
        return None, None

    h, w = mask.shape
    pts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pts = pts[0][0].reshape((-1, 2))

    if (pts[:, 0] < boundrary_thres).any() or (pts[:, 0] > resolution_height - boundrary_thres).any() \
            or (pts[:, 1] < boundrary_thres).any() or (pts[:, 1] > resolution_width - boundrary_thres).any():
        return None, None

    rgb = np.zeros((h, w, 3)).astype('uint8')

    depth_min = np.zeros((h, w)).astype('float32')
    depth_min[pts[:, 1], pts[:, 0]] = min_d
    depth_max = np.zeros((h, w)).astype('float32')
    depth_max[pts[:, 1], pts[:, 0]] = max_d

    pcd_crop = load_pcd_from_rgbd(rgb, depth_min, intrinsic, extrinsic)
    pcd_crop += load_pcd_from_rgbd(rgb, depth_max, intrinsic, extrinsic)

    hull, _ = pcd_crop.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))

    hull = o3d.geometry.BoundingConvexHull(pcd_crop)
    return hull, hull_ls


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
    # parser.add_argument('--scene_name', default='scene_2211192313')
    parser.add_argument('--scene_name', default='scene_2210232307_01')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--hand_mask_dir', default='hand_pose/d2/mask')

    args = parser.parse_args()
    scene_name = args.scene_name
    start_image_num = args.start_frame
    hand_mask_dir = args.hand_mask_dir

    gui.Application.instance.initialize()
    # PointCloudWindow(scene_name, start_image_num, hand_mask_dir=hand_mask_dir, obj_pose_file='object_pose/ground_truth.csv')
    w = PointCloudWindow(scene_name, start_image_num, hand_mask_dir=hand_mask_dir, obj_pose_file='object_pose/multiview_medium/object_poses.csv')
    # w.save_hand_obj_dists(f'{dataset_path}/{scene_name}/segmentation_points/point_cloud/obj_states.csv')

    gui.Application.instance.run()



if __name__ == "__main__":
    main()
