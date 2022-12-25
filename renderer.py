import glob
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import time

import cv2
import numpy as np
import pyrender
import trimesh
from tqdm import tqdm

from dataset_tools.bop_toolkit.bop_toolkit_lib import renderer
from dataset_tools.config import dataset_path
from dataset_tools.loaders import load_intrinsics, get_camera_names, load_object_pose_table
from dataset_tools.view.preview_videos import combine_videos

models_path = f'{dataset_path}/models'
obj_model_paths = {1: f'{models_path}/002_master_chef_can/textured_simple.obj',
                   2: f'{models_path}/003_cracker_box/textured_simple.obj',
                   3: f'{models_path}/004_sugar_box/textured_simple.obj',
                   4: f'{models_path}/005_tomato_soup_can/textured_simple.obj',
                   5: f'{models_path}/006_mustard_bottle/textured_simple.obj',
                   6: f'{models_path}/007_tuna_fish_can/textured_simple.obj',
                   7: f'{models_path}/008_pudding_box/textured_simple.obj',
                   8: f'{models_path}/009_gelatin_box/textured_simple.obj',
                   9: f'{models_path}/010_potted_meat_can/textured_simple.obj',
                   10: f'{models_path}/011_banana/textured_simple.obj',
                   11: f'{models_path}/019_pitcher_base/textured_simple.obj',
                   12: f'{models_path}/021_bleach_cleanser/textured_simple.obj',
                   13: f'{models_path}/024_bowl/textured_simple.obj',
                   14: f'{models_path}/025_mug/textured_simple.obj',
                   15: f'{models_path}/035_power_drill/textured_simple.obj',
                   16: f'{models_path}/036_wood_block/textured_simple.obj',
                   17: f'{models_path}/037_scissors/textured_simple.obj',
                   18: f'{models_path}/040_large_marker/textured_simple.obj',
                   19: f'{models_path}/051_large_clamp/textured_simple.obj',
                   20: f'{models_path}/052_extra_large_clamp/textured_simple.obj',
                   21: f'{models_path}/061_foam_brick/textured_simple.obj',
                   22: f'{models_path}/026_sponge/textured_simple.obj'}

ply_models_path = f'{dataset_path}/ply_models'
ply_model_paths = {1: f'{ply_models_path}/obj_000001.ply',
                   2: f'{ply_models_path}/obj_000002.ply',
                   3: f'{ply_models_path}/obj_000003.ply',
                   4: f'{ply_models_path}/obj_000004.ply',
                   5: f'{ply_models_path}/obj_000005.ply',
                   6: f'{ply_models_path}/obj_000006.ply',
                   7: f'{ply_models_path}/obj_000007.ply',
                   8: f'{ply_models_path}/obj_000008.ply',
                   9: f'{ply_models_path}/obj_000009.ply',
                   10: f'{ply_models_path}/obj_000010.ply',
                   11: f'{ply_models_path}/obj_000011.ply',
                   12: f'{ply_models_path}/obj_000012.ply',
                   13: f'{ply_models_path}/obj_000013.ply',
                   14: f'{ply_models_path}/obj_000014.ply',
                   15: f'{ply_models_path}/obj_000015.ply',
                   16: f'{ply_models_path}/obj_000016.ply',
                   17: f'{ply_models_path}/obj_000017.ply',
                   18: f'{ply_models_path}/obj_000018.ply',
                   19: f'{ply_models_path}/obj_000019.ply',
                   20: f'{ply_models_path}/obj_000020.ply',
                   21: f'{ply_models_path}/obj_000021.ply',
                   22: '/home/gdk/Data/kitchen_countertops/models/026_sponge/textured_simple.obj.ply'}

models_info_path = '/media/gdk/Data/Datasets/bop_datasets/ycbv/models_eval/models_info.json'

model_names = [
     'master_chef_can',
     'cracker_box',
     'sugar_box',
     'tomato_soup_can',
     'mustard_bottle',
     'tuna_fish_can',
     'pudding_box',
     'gelatin_box',
     'potted_meat_can',
     'banana',
     'pitcher_base',
     'bleach_cleanser',
     'bowl',
     'mug',
     'power_drill',
     'wood_block',
     'scissors',
     'large_marker',
     'large_clamp',
     'extra_large_clamp',
     'foam_brick',
     'sponge'
]


def overlay_imgs(im1, im2, p1=0.5, p2=0.5):
    im = p1 * im1.astype(np.float32) + p2 * im2.astype(np.float32)
    im[im > 255] = 255
    im = im.astype(np.uint8)
    return im


def load_meshes(obj_ids=None):
    meshes = {}
    if obj_ids == 'all':
        print('Loading all ycb models... (This may take 30 seconds)')
        for id, model_path in obj_model_paths.items():
            mesh = trimesh.load(model_path)
            mesh = pyrender.Mesh.from_trimesh(mesh)
            meshes[id] = mesh
    elif obj_ids is not None:
        for id in obj_ids:
            mesh = trimesh.load(obj_model_paths[id])
            mesh = pyrender.Mesh.from_trimesh(mesh)
            meshes[id] = mesh
    return meshes


def create_scene(intrinsics, obj_ids=None, pre_load_meshes=None):
    # Create pyrender scene.
    print('Creating pyrender...')
    scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]), ambient_light=np.array([1.0, 1.0, 1.0]))

    # Add camera.
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

    meshes = pre_load_meshes if pre_load_meshes else load_meshes(obj_ids)

    return (scene, meshes, cam)


def render_obj_pose(renderer, dict_id_poses=None, list_id_pose=None, width=640, height=480, unit='mm'):
    scene, meshes, cam = renderer
    scene.add(cam, pose=np.eye(4))

    if list_id_pose is None:
        list_id_pose = []
        for id, poses in dict_id_poses.items():
            for pose in poses:
                list_id_pose.append((id, pose))

    for id, pose in list_id_pose:
        if np.all(pose == 0.0):
            continue
        pose_copy = pose.copy()
        if len(pose) == 3:
            pose_copy = np.vstack((pose, np.array([[0, 0, 0, 1]], dtype=np.float32)))
        pose_copy[1] *= -1
        pose_copy[2] *= -1
        if unit == 'mm':
            pose_copy[:3, 3] /= 1000.0

        if id not in meshes:
            new_meshes = load_meshes([id])
            meshes[id] = new_meshes[id]

        scene.add(meshes[id], pose=pose_copy)

    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    rendered_im, _ = r.render(scene)
    scene.clear()
    return rendered_im[:, :, ::-1]


def render_points(renderer, xyzs, width=640, height=480, unit='m'):
    scene, meshes, cam = renderer
    scene.add(cam, pose=np.eye(4))

    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.0, 1.0, 0.0]
    m = pyrender.Mesh.from_trimesh(sm)

    for xyz in xyzs:
        pose = np.eye(4)
        pose[:3, -1] = xyz / 1000.0 if unit == 'mm' else xyz
        pose[1] *= -1
        pose[2] *= -1
        scene.add(m, pose=pose)

    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    rendered_im, _ = r.render(scene)
    scene.clear()
    return rendered_im


def compare_gt_est(gt_im, est_im):
    # cv2.imshow('gt', gt_im)
    # cv2.imshow('est', est_im)
    gt_im_green = np.zeros_like(gt_im)
    est_im_red = np.zeros_like(est_im)
    gt_im_green[:, :, 1] = cv2.cvtColor(gt_im, cv2.COLOR_BGR2GRAY)
    est_im_red[:, :, 2] = cv2.cvtColor(est_im, cv2.COLOR_BGR2GRAY)
    comp = overlay_imgs(gt_im_green, est_im_red, 3, 3)
    # cv2.imshow('comp', comp)
    # cv2.imshow('green', gt_im_green)
    # cv2.imshow('red', est_im_red)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return comp


def render_obj_pose_table(camera_path, pose_path, save_dir):
    out = cv2.VideoWriter(f'{save_dir}/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

    intrinsics = load_intrinsics(f'{camera_path}/camera_meta.yml')
    py_renderer = create_scene(intrinsics)
    color_img_paths = sorted(glob.glob(f'{camera_path}/rgb/*.png'))
    opt = load_object_pose_table(pose_path)

    for frame, image_path in enumerate(tqdm(color_img_paths)):
        im = cv2.imread(image_path)

        list_id_pose = opt[opt['frame'] == frame][['obj_id', 'pose']]
        rendered_im = render_obj_pose(py_renderer, list_id_pose=list_id_pose)
        out_im = overlay_imgs(im, rendered_im)

        cv2.imwrite(f'{save_dir}/{frame:06d}.png', out_im)
        out.write(out_im)

    out.release()


def render_scene(scene_name, save_folder_name, opt_name):
    scene_path = f'{dataset_path}/{scene_name}'
    for camera_name in get_camera_names(scene_path):
        print(camera_name)
        camera_path = f'{dataset_path}/{scene_name}/{camera_name}'
        save_dir = f'{camera_path}/{save_folder_name}'
        render_obj_pose_table(camera_path, f'{save_dir}/{opt_name}', save_dir)
    video_paths = sorted(glob.glob(f'{scene_path}/camera_*/{save_folder_name}/video.mp4'))
    save_path = f'{scene_path}/{save_folder_name}/video.mp4'
    combine_videos(video_paths, save_path)


if __name__ == '__main__':
    # # load gt
    # gt_path = '/home/gdk/data/scene_220603104027/camera_03/scene_gt.json'
    # gt = load_ground_truth(gt_path)
    # dict_id_poses = gt[0]
    # print(dict_id_poses)
    # im_path = '/home/gdk/data/scene_220603104027/camera_03/rgb/000000.png'
    # intrinsics = load_intrinsics('/home/gdk/data/scene_220603104027/camera_03/camera_meta.yml')
    # py_renderer = create_scene(intrinsics, dict_id_poses.keys())
    # py_rendered_im = render_obj_pose(py_renderer, dict_id_poses, unit='mm')
    # py_rendered_im = overlay_imgs(cv2.imread(im_path), py_rendered_im)
    # cv2.imshow('gt', py_rendered_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()

    # test comp gt and est
    # gt_im = cv2.imread('gt_screenshot_01.10.2022.png')
    # est_im = cv2.imread('est_screenshot_01.10.2022.png')
    # compare_gt_est(gt_im, est_im)
    # exit()


    # Test render objs
    test_path = '/media/gdk/Hard Disk/Datasets/old_kitchen_data/multiple_depth_cameras/1652915948/752112070756'
    test_img_name = '000036_color.png'
    dict_id_poses = {3: [np.array([[0.50324144, 0.86231973, -0.05614924, -0.00303771],
                                   [0.78285323, -0.48244995, -0.39291584, -0.0212927],
                                   [-0.36590828, 0.15377491, -0.9178586, 1.0443474]])]}

    intrinsics = load_intrinsics(f'{test_path}/meta.yml')
    test_im = cv2.imread(f'{test_path}/{test_img_name}')

    # render objs with pyrender
    py_renderer = create_scene(intrinsics, dict_id_poses.keys())
    t = time.time()
    py_rendered_im = render_obj_pose(py_renderer, dict_id_poses)
    print(time.time() - t)
    py_rendered_im = overlay_imgs(test_im, py_rendered_im)

    # render objs with vispy
    h, w, _ = test_im.shape
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    vi_renderer = renderer.create_renderer(w, h, renderer_type='vispy', mode='rgb')

    for obj_id in dict_id_poses.keys():
        vi_renderer.add_object(obj_id, ply_model_paths[obj_id])

    vi_rendered_im = np.zeros_like(test_im)
    for obj_id, poses in dict_id_poses.items():
        for pose in poses:
            t = time.time()
            out_im = vi_renderer.render_object(obj_id, pose[:3, :3], pose[:3, 3] * 1000, fx, fy, cx, cy)['rgb']
            print(time.time() - t)

            out_im = out_im[:, :, ::-1]
            vi_rendered_im = overlay_imgs(vi_rendered_im, out_im, 1, 1)
    vi_rendered_im = overlay_imgs(test_im, vi_rendered_im)

    # show imgs
    cv2.imshow('original', test_im)
    cv2.imshow('pyrender', py_rendered_im)
    cv2.imshow('vispy', vi_rendered_im)
    while cv2.waitKey(5) < 0:  # Press any key to load subsequent image
        continue
    cv2.destroyAllWindows()

    # -----------------
    # test pts render
    test_path = '/media/gdk/Hard Disk/Datasets/old_kitchen_data/multiple_depth_cameras/1652915948/752112070756'
    test_img_name = '000036_color.png'
    xyzs = [[-0.00303771, -0.0212927, 1.0443474]]

    intrinsics = load_intrinsics(f'{test_path}/meta.yml')
    test_im = cv2.imread(f'{test_path}/{test_img_name}')

    # pyrender pts
    py_renderer = create_scene(intrinsics)
    py_rendered_im = render_points(py_renderer, xyzs)
    py_rendered_im = overlay_imgs(test_im, py_rendered_im)

    cv2.imshow('original', test_im)
    cv2.imshow('pts', py_rendered_im)
    while cv2.waitKey(5) < 0:  # Press any key to load subsequent image
        continue
    cv2.destroyAllWindows()

