import cv2
import numpy as np
import open3d as o3d

from dataset_tools.loaders import load_intrinsics, load_extrinsics, get_depth_scale
from dataset_tools.view.point_cloud_simple import load_pcd_from_rgbd

mask = cv2.imread('/home/gdk/Data/kitchen_countertops/scene_2210232307_01/camera_01_943222070458/hand_pose/d2/mask/000017.png', 0)

pts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

p = pts[0][0].reshape((-1, 2))

rgb = np.zeros((480, 640, 3)).astype('uint8')
# rgb[p[:, 0], p[:, 1], :] = 255

depth_close = np.zeros((480, 640)).astype('float32')
depth_close[p[:, 1], p[:, 0]] = 0.1
depth_far = np.zeros((480, 640)).astype('float32')
depth_far[p[:, 1], p[:, 0]] = 2

intrinsic = load_intrinsics('/home/gdk/Data/kitchen_countertops/scene_2210232307_01/camera_01_943222070458/camera_meta.yml')
extrinsic = load_extrinsics('/home/gdk/Data/kitchen_countertops/scene_2210232307_01/extrinsics.yml')['camera_01_943222070458']
pcd_crop = load_pcd_from_rgbd(rgb, depth_close, intrinsic, extrinsic)
pcd_crop += load_pcd_from_rgbd(rgb, depth_far, intrinsic, extrinsic)

hull, _ = pcd_crop.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))
# o3d.visualization.draw_geometries([pcd, hull_ls])

print(np.asarray(pcd_crop.points))

# volume = o3d.visualization.SelectionPolygonVolume()
# volume.orthogonal_axis = "Z"
# volume.axis_max = 1000
# volume.axis_min = -1000
# volume.bounding_polygon = o3d.utility.Vector3dVector(pcd.points)

rgb_path = '/home/gdk/Data/kitchen_countertops/scene_2210232307_01/camera_01_943222070458/rgb/000017.png'
rgb_img = cv2.imread(rgb_path)
depth_path = '/home/gdk/Data/kitchen_countertops/scene_2210232307_01/camera_01_943222070458/depth/000017.png'
depth_img = cv2.imread(depth_path, -1)
depth_scale = get_depth_scale('/home/gdk/Data/kitchen_countertops/scene_2210232307_01/camera_01_943222070458/camera_meta.yml', convert2unit='m')
depth_img = np.float32(depth_img * depth_scale)
pcd = load_pcd_from_rgbd(rgb_img, depth_img, intrinsic, extrinsic)

hull = o3d.geometry.BoundingConvexHull(pcd_crop)
pcd = pcd.crop_convex_hull(hull)

o3d.visualization.draw_geometries([pcd, hull_ls])

# cv2.imshow('t', mask)
# cv2.waitKey(0)