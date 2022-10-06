import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import cv2
import yaml
import pyrender
import trimesh

from bop_toolkit_lib import renderer
from config import obj_model_paths, ply_model_paths
from modules.helpers.loaders import load_intrinsics


def overlay_imgs(im1, im2, p1=0.5, p2=0.5):
    im = p1 * im1.astype(np.float32) + p2 * im2.astype(np.float32)
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


def render_obj_pose(renderer, dict_id_poses, width=640, height=480, unit='m'):
    scene, meshes, cam = renderer
    scene.add(cam, pose=np.eye(4))

    for id, poses in dict_id_poses.items():
        for pose in poses:
            if np.all(pose == 0.0):
                continue
            pose_copy = pose.copy()
            if len(pose) == 3:
                pose_copy = np.vstack((pose, np.array([[0, 0, 0, 1]], dtype=np.float32)))
            pose_copy[1] *= -1
            pose_copy[2] *= -1
            if unit == 'mm':
                pose_copy[:3, 3] /= 1000.0
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


if __name__ == '__main__':
    # test comp gt and est
    # gt_im = cv2.imread('gt_screenshot_01.10.2022.png')
    # est_im = cv2.imread('est_screenshot_01.10.2022.png')
    # compare_gt_est(gt_im, est_im)
    # exit()


    # Test render objs
    test_path = '/home/gdk/data/1652915948/752112070756'
    test_img_name = '000036_color.png'
    dict_id_poses = {3: [np.array([[0.50324144, 0.86231973, -0.05614924, -0.00303771],
                                   [0.78285323, -0.48244995, -0.39291584, -0.0212927],
                                   [-0.36590828, 0.15377491, -0.9178586, 1.0443474]])]}

    intrinsics = load_intrinsics(f'{test_path}/meta.yml')
    test_im = cv2.imread(f'{test_path}/{test_img_name}')

    # render objs with pyrender
    py_renderer = create_scene(intrinsics, dict_id_poses.keys())
    py_rendered_im = render_obj_pose(py_renderer, dict_id_poses)
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
            out_im = vi_renderer.render_object(obj_id, pose[:3, :3], pose[:3, 3] * 1000, fx, fy, cx, cy)['rgb']
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
    test_path = '/home/gdk/data/1652915948/752112070756'
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
