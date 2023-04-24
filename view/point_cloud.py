import os

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import pandas as pd
from tqdm import tqdm

from dataset_tools.config import dataset_path, resolution_width, resolution_height
from dataset_tools.loaders import get_num_frame
from dataset_tools.view.open3d_window import Open3dWindow


class PointCloudWindow(Open3dWindow):
    def __init__(self, scene_name, init_frame_num, width=640 * 3 + 408, height=480 * 3, hand_mask_dir=None,
                 obj_pose_file=None, infra_pose_file=None):
        super().__init__(width, height, scene_name, 'Point Cloud', init_frame_num, obj_pose_file, infra_pose_file)
        self.bounds = None
        self.frame_num = init_frame_num
        self.update_frame_label()
        em = self.window.theme.font_size

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)
        self.scene_widget.set_on_key(self.on_keyboard_input)

        self.window.set_on_layout(self.on_layout)

        # mask
        self.hand_mask_dir = hand_mask_dir
        self.hand_mask_box = gui.Checkbox(f'Apply Masks')
        if hand_mask_dir is not None:
            self.hand_masks = {}
            self.view_ctrls.add_child(gui.Label(hand_mask_dir))
            self.hand_mask_box.set_on_checked(self.update_frame)
            self.view_ctrls.add_child(self.hand_mask_box)

            # intersection
            self.hand_mask_intsct_box = gui.Checkbox(f'Intersect Masks')
            self.hand_mask_intsct_box.set_on_checked(self.update_frame)
            self.view_ctrls.add_child(self.hand_mask_intsct_box)
            self.hand_mask_intsct_convex_hull_box = gui.Checkbox(f'Show convex hull')
            self.hand_mask_intsct_convex_hull_box.set_on_checked(self.update_frame)
            self.view_ctrls.add_child(self.hand_mask_intsct_convex_hull_box)
            self.view_ctrls.add_child(gui.Label(""))

        # select camera views
        self.view_ctrls.add_child(gui.Label("Active Cameras"))
        self.selected_cameras = []
        grid = gui.VGrid(2, 0.25 * em)
        self.view_ctrls.add_child(grid)
        for i in [0, 4, 1, 5, 2, 6, 3, 7]:
            box = gui.Checkbox(f'camera {i + 1:02d}')
            box.checked = True
            box.set_on_checked(self.update_frame)
            grid.add_child(box)
            self.selected_cameras.append((i, box))

        self.update_frame()
        self.set_active_camera_view()

    def on_layout(self, layout_context):
        r = self.window.content_rect
        width = 17 * layout_context.theme.font_size
        height = max(r.height,
                     self.settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        self.settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)
        self.scene_widget.frame = gui.Rect(0, r.y, r.get_right() - width, r.height)

    def on_point_size(self, size):
        self.settings.pcd_material.point_size = int(size)
        self.scene_widget.scene.modify_geometry_material("pcd", self.settings.pcd_material)

    def set_active_camera_view(self):
        camera_name = self.camera_names[self.active_camera_view]
        intrinsic = self.intrinsics[camera_name]
        extrinsic = self.extrinsics[camera_name]
        self.scene_widget.setup_camera(intrinsic, extrinsic, 640, 480, self.bounds)

    def load_pcds(self):
        pcds = o3d.geometry.PointCloud()
        for camera_name in self.get_selected_camera_names():
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

    def load_convex_hulls_ls(self):
        hulls = []
        hull_lss = []
        for camera_name in self.get_selected_camera_names():
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

    def update_frame(self, arg=None):
        self.scene_widget.scene.clear_geometry()
        self.update_frame_label()
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
            for obj_id in self.opt[self.opt['frame'] == self.frame_num]['obj_id']:
                mesh = self.load_obj_mesh(obj_id)
                self.scene_widget.scene.add_geometry(str(obj_id), mesh, self.settings.obj_material)

        if self.infra_pose_box.checked:
            for obj_id in self.ipt['obj_id']:
                mesh = self.load_infra_mesh(obj_id)
                self.scene_widget.scene.add_geometry(str(obj_id), mesh, self.settings.obj_material)

    def get_selected_camera_names(self):
        active_camera_ids = []
        for i, box in self.selected_cameras:
            if box.checked:
                active_camera_ids.append(i)
        return np.array(self.camera_names)[active_camera_ids]

    def on_keyboard_input(self, event):
        if event.is_repeat:
            return gui.Widget.EventCallbackResult.HANDLED

        # Change camera view
        if 49 <= event.key <= 56:
            if event.type == gui.KeyEvent.DOWN:
                view_id = event.key - 49
                if view_id < len(self.camera_names):
                    self.on_change_active_camera_view(view_id)
                    return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.HANDLED

    def save_distances(self, save_file_path, dist_tres=0.01, valid_infra_ids=[101]):
        detections = np.empty(0, dtype=[('frame', 'i4'),
                                        ('obj_id_1', 'i4'),
                                        ('obj_id_2', 'i4'),
                                        ('relationship', 'U20'),
                                        ('in_contact', 'i1'),
                                        ('distance', 'f')])

        for frame in tqdm(range(get_num_frame(self.scene_path))):
            self.frame_num = frame
            self.load_images()
            self.load_hand_masks()
            pcd = self.load_pcds()
            convex_hulls, hull_lss = self.load_convex_hulls_ls()
            hand_pcd = crop_pcd_by_convex_hulls(pcd, convex_hulls)

            for obj_id in self.opt[self.opt['frame'] == frame]['obj_id']:
                obj_mesh = self.load_obj_mesh(obj_id)
                obj_pcd = obj_mesh.sample_points_uniformly(1000)
                # obj-hand
                if len(convex_hulls) > 0:
                    dist = compute_pcds_dist(hand_pcd, obj_pcd)
                    detections = np.append(detections,
                                           np.array([(frame, obj_id, 0, 'obj-hand', 0, dist)], dtype=detections.dtype))

                # obj-infra
                for infra_id in valid_infra_ids:
                    if infra_id in self.ipt['obj_id']:
                        infra_mesh = self.load_infra_mesh(infra_id)
                        infra_pcd = infra_mesh.sample_points_uniformly(1000)
                        dist = compute_pcds_dist(infra_pcd, obj_pcd)
                        detections = np.append(detections,
                                               np.array([(frame, obj_id, infra_id, 'obj-infra', 0, dist)],
                                                        dtype=detections.dtype))

            # obj-infra
            for infra_id in valid_infra_ids:
                if infra_id in self.ipt['obj_id'] and len(convex_hulls) > 0:
                    infra_mesh = self.load_infra_mesh(infra_id)
                    infra_pcd = infra_mesh.sample_points_uniformly(1000)
                    dist = compute_pcds_dist(infra_pcd, hand_pcd)
                    detections = np.append(detections,
                                           np.array([(frame, infra_id, 0, 'infra-hand', 0, dist)],
                                                    dtype=detections.dtype))

        detections['in_contact'][detections['distance'] < dist_tres] = 1

        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        df = pd.DataFrame.from_records(detections)
        df = df.sort_values(by=['obj_id_1', 'obj_id_2', 'frame'])
        df.to_csv(save_file_path, index=False)
        print(f'distances saved to {save_file_path}')


def compute_pcds_dist(pcd1, pcd2):
    return min(pcd1.compute_point_cloud_distance(pcd2))


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


def load_pcd_from_rgbd(rgb_img, depth_img, intrisic, extrinsic):
    rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    depth_img_o3d = o3d.geometry.Image(depth_img)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_img.shape[0], rgb_img.shape[1],
                                                  intrisic[0, 0], intrisic[1, 1], intrisic[0, 2], intrisic[1, 2])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d,
                                                              depth_scale=1, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
    return pcd


def main():
    scene_name = 'scene_230313172100'
    start_image_num = 0
    hand_mask_dir = 'hand_pose/d2/mask'
    # obj_pose_file = 'object_pose/multiview_medium_smooth/object_poses.csv'
    obj_pose_file = '../object_pose/ground_truth.csv'
    infra_pose_file = 'infra_poses.csv'

    gui.Application.instance.initialize()
    w = PointCloudWindow(scene_name, start_image_num,
                         hand_mask_dir=hand_mask_dir, obj_pose_file=obj_pose_file, infra_pose_file=infra_pose_file)
    w.save_distances(f'{dataset_path}/{scene_name}/segmentation_points/point_cloud/obj_states.csv')

    gui.Application.instance.run()


if __name__ == "__main__":
    main()
