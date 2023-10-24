import glob
import os

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import pandas as pd
from tqdm import tqdm

from dataset_tools.config import dataset_path, resolution_width, resolution_height
from dataset_tools.utils.name import get_newest_scene_name
from dataset_tools.view.open3d_window import Open3dWindow


class PointCloudWindow(Open3dWindow):
    def __init__(self, scene_name, init_frame_num, width=640 * 3 + 408, height=480 * 3, mask_dir=None,
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
        self.mask_dir = mask_dir
        self.mask_box = gui.Checkbox(f'Apply Masks')
        self.masks = {}
        self.mask_intsct_box = gui.Checkbox(f'Intersect Masks')
        if mask_dir is not None:
            self.view_ctrls.add_child(gui.Label(mask_dir))
            self.mask_box.set_on_checked(self.update_frame)
            self.view_ctrls.add_child(self.mask_box)

            # intersection
            self.mask_intsct_box.set_on_checked(self.update_frame)
            self.view_ctrls.add_child(self.mask_intsct_box)
            self.mask_intsct_convex_hull_box = gui.Checkbox(f'Show convex hull')
            self.mask_intsct_convex_hull_box.set_on_checked(self.update_frame)
            self.view_ctrls.add_child(self.mask_intsct_convex_hull_box)
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

    def load_pcds(self, frame_num=None, mask_dir=None, selected_cameras=None):
        if frame_num is not None:
            self.frame_num = frame_num
        if mask_dir is not None:
            self.mask_dir = mask_dir
            self.mask_box.checked = True
        if selected_cameras is not None:
            for i, box in self.selected_cameras:
                box.checked = True if i in selected_cameras else False
        if frame_num is not None or mask_dir is not None or selected_cameras is not None:
            self.update_frame()

        pcds = o3d.geometry.PointCloud()
        for camera_name in self.get_selected_camera_names():
            intrinsic = self.intrinsics[camera_name]
            extrinsic = self.extrinsics[camera_name]
            rgb_img = self.rgb_imgs[camera_name]
            depth_img = self.depth_imgs[camera_name]

            if self.mask_box.checked:
                mask = self.masks[camera_name]
                rgb_img = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
                depth_img = cv2.bitwise_and(depth_img, depth_img, mask=mask)

            pcds += load_pcd_from_rgbd(rgb_img, depth_img, intrinsic, extrinsic)
        return pcds

    def load_masks(self):
        for camera_name in self.camera_names:
            mask_path = f'{self.scene_path}/{camera_name}/{self.mask_dir}/{self.frame_num:06}.png'
            if os.path.exists(mask_path):
                self.masks[camera_name] = cv2.imread(mask_path, 0)
            else:
                self.masks[camera_name] = np.zeros((resolution_height, resolution_width)).astype('uint8')

    def load_convex_hulls_ls(self):
        hulls = []
        hull_lss = []
        for camera_name in self.get_selected_camera_names():
            intrinsic = self.intrinsics[camera_name]
            extrinsic = self.extrinsics[camera_name]
            mask = self.masks[camera_name]
            if mask.max() == 0:
                continue
            hull, hull_ls = compute_convex_hull_line_set_from_mask(mask, intrinsic, extrinsic)
            if hull is not None:
                hulls.append(hull)
                hull_lss.append(hull_ls)
        return hulls, hull_lss

    def update_frame(self, frame=None):
        super().update_frame(frame)
        self.scene_widget.scene.clear_geometry()
        self.update_frame_label()
        self.load_images()
        self.load_masks()
        pcd = self.load_pcds()

        if self.mask_intsct_box.checked:
            convex_hulls, hull_lss = self.load_convex_hulls_ls()
            pcd = crop_pcd_by_convex_hulls(pcd, convex_hulls)
            if self.mask_intsct_convex_hull_box.checked:
                for i, hull_ls in enumerate(hull_lss):
                    self.scene_widget.scene.add_geometry(f"hull_{i}", hull_ls, self.settings.line_set_material)

        self.bounds = pcd.get_axis_aligned_bounding_box()
        self.scene_widget.scene.add_geometry("pcd", pcd, self.settings.pcd_material)

        # add objs
        if self.obj_pose_box.checked:
            for object_name in np.unique(self.opt.table['object_name']):
                mesh = self.load_obj_mesh(object_name)
                if mesh is None:
                    continue
                self.scene_widget.scene.add_geometry(str(object_name), mesh, self.settings.obj_material)

    def get_selected_camera_names(self):
        active_camera_ids = []
        for i, box in self.selected_cameras:
            if box.checked:
                active_camera_ids.append(i)
        return np.array(self.camera_names)[active_camera_ids]


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
    # scene_name = 'scene_231010164827_exe'
    scene_name = get_newest_scene_name()
    start_image_num = 220356
    # mask_dir = 'hand_pose/d2/mask'
    # mask_dir = 'masks/bowl'
    mask_dir = None
    # obj_pose_file = 'object_pose/multiview_medium/object_poses.csv'
    # obj_pose_file = '../object_pose/point_cloud.csv'
    # obj_pose_file = '../object_pose/ground_truth.csv'
    # obj_pose_file = '../object_pose_table.csv'
    obj_pose_file = None
    # infra_pose_file = 'infra_poses.csv'
    infra_pose_file = None

    gui.Application.instance.initialize()
    w = PointCloudWindow(scene_name, start_image_num,
                         mask_dir=mask_dir,
                         obj_pose_file=obj_pose_file,
                         infra_pose_file=infra_pose_file)
    # w.save_distances(f'{dataset_path}/{scene_name}/segmentation_points/point_cloud/obj_states.csv')

    gui.Application.instance.run()


if __name__ == "__main__":
    main()
