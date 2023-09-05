import csv
import os
from pprint import pprint

import cv2
import numpy as np
from tqdm import tqdm

from dataset_tools.config import dataset_path
from dataset_tools.utils import load_cameras_intrisics, load_cameras_extrinsics, load_imgs_across_cameras, \
    get_camera_names, intr2param, load_object_pose_table, load_infra_pose, save_object_pose_table, get_available_frames
from dataset_tools.record.apriltag_detection import detect_april_tag, draw_pose, draw_pose_axes
from dataset_tools.view.helpers import collage_imgs
from modules.object_pose.multiview_voting import combine_poses


def get_tag_poses(scene_name, frames, target_tag_id, tag_size, mode='best'):
    print('Extracting tag pose...')

    scene_path = f'{dataset_path}/{scene_name}'

    cameras_intr = load_cameras_intrisics(scene_name)
    cameras_ext = load_cameras_extrinsics(scene_name)

    if mode == 'best':
        min_ef = np.Inf
        best_pose = None
        for frame in tqdm(frames, disable=True if len(frames) < 10 else False):
            imgs = load_imgs_across_cameras(scene_path, get_camera_names(scene_path), f'{frame:06d}.png')
            for camera, img in zip(get_camera_names(scene_path), imgs):
                intr = cameras_intr[camera]
                param = intr2param(intr)
                poses, overlay = detect_april_tag(img, param, tag_size)
                for tag_id, pose, ef in poses:
                    if tag_id == target_tag_id and ef < min_ef:
                        min_ef = ef
                        best_pose = (camera, pose)
        camera, pose = best_pose
    elif mode == 'combine':
        poses = []
        c0 = get_camera_names(scene_path)[0]
        for frame in tqdm(frames, disable=True if len(frames) < 10 else False):
            imgs = load_imgs_across_cameras(scene_path, get_camera_names(scene_path), f'{frame:06d}.png')
            for camera, img in zip(get_camera_names(scene_path), imgs):
                intr = cameras_intr[camera]
                param = intr2param(intr)
                _poses, overlay = detect_april_tag(img, param, tag_size)
                _poses = [pose for tag_id, pose, ef in _poses if tag_id == target_tag_id]
                _poses = [np.linalg.inv(cameras_ext[c0]) @ cameras_ext[camera] @ pose for pose in _poses]
                poses.extend(_poses)
        pose = combine_poses(poses, thres_dist=0.01, thres_q_sim=0.99, verbose=True)
        camera = c0
    else:
        raise ValueError(f'Unknown mode: {mode}')

    # change to right pose
    rm = np.zeros((4, 4))
    rm[0, 0] = 1
    rm[1, 1] = -1
    rm[2, 2] = -1
    rm[3, 3] = 1
    tag_pose = pose @ rm

    rm = np.zeros((4, 4))
    rm[0, 1] = 1
    rm[1, 0] = -1
    rm[2, 2] = 1
    rm[3, 3] = 1
    tag_pose = tag_pose @ rm

    return cameras_ext[camera] @ tag_pose


def extract_infra_pose(scene_name, infra_tag_id=1, infra_tag_size=0.06, num_frames=10):
    tag_pose = get_tag_poses(scene_name, get_available_frames(scene_name)[:num_frames], infra_tag_id, infra_tag_size)

    record_infra_pose(scene_name, 'sink_unit', tag_pose)

    t = np.array([[0, 0, 1, 0.47793],
                  [1, 0, 0, -0.06707],
                  [0, 1, 0, -0.29],
                  [0, 0, 0, 1]])
    sink_only_pose = tag_pose @ t
    record_infra_pose(scene_name, 'sink_only', sink_only_pose)


def record_infra_pose(scene_name, obj_name, pose):
    # TODO: hardcoded
    if obj_name == 'sink_unit':
        obj_id = 100
    elif obj_name == 'sink_only':
        obj_id = 101
    elif obj_name == 'robot_tag':
        obj_id = 110
    elif obj_name == 'robot_base':
        obj_id = 111
    elif obj_name == 'blue_tip':
        obj_id = 120
    else:
        raise ValueError(f'unknown object name {obj_name}')

    ipt_path = f'{dataset_path}/{scene_name}/infra_poses.csv'
    if os.path.exists(ipt_path):
        print(f'Updating {obj_name} pose in existing infra_poses.csv...')
        ipt = load_object_pose_table(ipt_path)
        if obj_id in ipt['obj_id']:
            ipt[(ipt['obj_id'] == obj_id).nonzero()[0][0]]['pose'] = pose
        else:
            ipt = np.append(ipt, np.array([(obj_id, obj_name, 0, pose)], dtype=ipt.dtype))
        save_object_pose_table(ipt, ipt_path)
    else:
        f = open(ipt_path, 'w')
        csv_writer = csv.writer(f)
        csv_writer.writerow(['obj_id', 'name', 'frame', 'pose'])
        csv_writer.writerow([obj_id, obj_name, 0, pose.tolist()])
        f.close()


def view_april_tag_pose(tag_pose, imgs, cameras_intr, camera_ext, tag_size, render_sugar_box=False):
    # original tags
    overlays = []
    for img, intr in zip(imgs, cameras_intr.values()):
        poses, overlay = detect_april_tag(img, intr2param(intr), tag_size)
        overlays.append(overlay)
    overlay = collage_imgs(overlays)
    cv2.imshow('orignal tags', overlay)

    for img, intr, ext in zip(imgs, cameras_intr.values(), camera_ext.values()):
        pose = np.linalg.inv(ext) @ tag_pose
        camera_params = intr2param(intr)
        draw_pose(img, camera_params, tag_size, pose)
        draw_pose_axes(img, camera_params, tag_size, pose)
    preview = collage_imgs(imgs)
    cv2.imshow('verify', preview)

    if render_sugar_box:
        from dataset_tools.view.renderer import create_renderer, render_obj_pose, set_intrinsics, overlay_imgs
        re_ims = []
        renderer = create_renderer()
        for img, intr, ext in zip(imgs, cameras_intr.values(), camera_ext.values()):
            pose = np.linalg.inv(ext) @ tag_pose

            set_intrinsics(renderer, intr)
            rendered_im = render_obj_pose(renderer, [(4, pose)], unit='m')
            rendered_im = overlay_imgs(img, rendered_im)
            re_ims.append(rendered_im)

            camera_params = intr2param(intr)
            draw_pose(img, camera_params, tag_size, pose)
            draw_pose_axes(img, camera_params, tag_size, pose)

        preview = collage_imgs(re_ims)
        cv2.imshow('verify using sugar box', preview)

    key = cv2.waitKey(0)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()


def view_infra_pose(scene_name, infra_name='sink_unit', frame_id=0, tag_size=0.06):
    scene_path = f'{dataset_path}/{scene_name}'
    imgs = load_imgs_across_cameras(scene_path, get_camera_names(scene_path), f'{frame_id:06d}.png')
    cameras_intr = load_cameras_intrisics(scene_name)
    cameras_ext = load_cameras_extrinsics(scene_name)
    infra_pose = load_infra_pose(scene_name, infra_name)
    view_april_tag_pose(infra_pose, imgs, cameras_intr, cameras_ext, tag_size=tag_size)


if __name__ == '__main__':
    scene_names = ['scene_230822164306']

    for scene_name in scene_names:
        extract_infra_pose(scene_name)
        view_infra_pose(scene_name, frame_id=get_available_frames(scene_name)[0], tag_size=0.06)
