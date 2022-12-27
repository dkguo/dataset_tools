import csv
import json
import os
import glob
import re

import cv2
import numpy as np
import pandas as pd
import yaml

from dataset_tools.config import dataset_path


def get_camera_names(scene_path):
    folders = os.listdir(scene_path)
    camera_seq = []
    for folder in folders:
        if folder[:7] == 'camera_' and os.path.isdir(f'{scene_path}/{folder}'):
            camera_seq.append(folder)
    return sorted(camera_seq)


def get_num_frame(scene_path):
    camera_names = get_camera_names(scene_path)
    nums = []
    for camera_name in camera_names:
        files = glob.glob(f'{scene_path}/{camera_name}/rgb/*.png')
        nums.append(len(files))
    return min(nums)


def save_mp4(imgs, video_save_path, frame_rate=15):
    assert video_save_path[-3:] == 'mp4', 'video_save_path has to end with mp4'
    dim = (imgs[0].shape[1], imgs[0].shape[0])
    print(f'Saving {video_save_path}')
    out_video = cv2.VideoWriter(f'{video_save_path}', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, dim)
    for im in imgs:
        out_video.write(im)
    out_video.release()


def save_imgs(imgs, save_path, uniqname=None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print(save_path, f'Saving {uniqname if uniqname is not None else ""}.png')
    for frame, im in enumerate(imgs):
        cv2.imwrite(f'{save_path}/{frame:06d}{uniqname if uniqname is not None else ""}.png', im)


def load_images(imgs_path, uniqname, mode=cv2.IMREAD_COLOR):
    imgs = []
    files = os.listdir(imgs_path)

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    files.sort(key=natural_keys)
    for f in files:
        if uniqname in f:
            im = cv2.imread(f'{imgs_path}/{f}', mode)
            imgs.append(im)
    return imgs


def load_imgs_across_cameras(scene_path, camera_names, image_name, mode=cv2.IMREAD_COLOR):
    imgs = []
    for camera_name in camera_names:
        im = cv2.imread(f'{scene_path}/{camera_name}/rgb/{image_name}', mode)
        imgs.append(im)
    return imgs


def load_intrinsics(camera_meta_path):
    with open(camera_meta_path, "r") as stream:
        intrinsics = np.array(yaml.safe_load(stream)['INTRINSICS']).reshape(3, 3)
    return intrinsics


def intr2param(intrinsics):
    """intrisics: 3x3"""
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    return fx, fy, cx, cy


def load_extrinsics(extrinsics_path, to_mm=False):
    with open(extrinsics_path, 'r') as file:
        ext = yaml.full_load(file)
    if 'EXTRINSICS' in ext:
        ext = ext['EXTRINSICS']
    for k, e in ext.items():
        ext[k] = np.array(e)
        if to_mm:
            ext[k][:3, 3] *= 1000.0
    return ext


def load_cameras_meta(scene_name):
    with open(f'{dataset_path}/{scene_name}/cameras_meta.yml') as f:
        return yaml.full_load(f)


def load_cameras_intrisics(scene_name):
    cameras_meta = load_cameras_meta(scene_name)
    cameras_intics = cameras_meta['INTRINSICS']
    for camera_name, intri in cameras_intics.items():
        cameras_intics[camera_name] = np.array(intri)
    return cameras_intics


def load_cameras_extrinsics(scene_name):
    cameras_meta = load_cameras_meta(scene_name)
    cameras_exts = cameras_meta['EXTRINSICS']
    for camera_name, ext in cameras_exts.items():
        cameras_exts[camera_name] = np.array(ext)
    return cameras_exts


def get_depth_scale(camera_meta_path, convert2unit='mm'):
    with open(camera_meta_path, "r") as stream:
        depth_unit = yaml.safe_load(stream)['DEPTH_UNIT']
    dict_unit_scale = {'m': 1000.0, 'mm': 1.0, 'μm': 0.1}
    depth_scale = dict_unit_scale[depth_unit] / dict_unit_scale[convert2unit]
    return depth_scale


def load_object_pose_table(file_path, only_valid_pose=False, fill_nan=False):
    """
    Returns:
        obj_pose_table (opt), numpy recarray
    """
    df = pd.read_csv(file_path, converters={'pose': lambda x: eval(f'np.array({x})')})
    if only_valid_pose:
        drop_idxs = []
        for i, pose in enumerate(df['pose']):
            if len(pose) != 4:
                drop_idxs.append(i)
        df = df.drop(index=drop_idxs)
    opt = df.to_records(index=False)

    if fill_nan:
        obj_ids = set(opt['obj_id'])
        scene_name = opt[0]['scene_name']
        for obj_id in obj_ids:
            opt_obj = opt[opt['obj_id'] == obj_id]
            for frame in range(get_num_frame(f'{dataset_path}/{scene_name}')):
                if frame not in opt_obj['frame']:
                    opt = np.append(opt, opt[-1])
                    opt[-1]['obj_id'] = obj_id
                    opt[-1]['frame'] = frame
                    opt[-1]['pose'] = np.full((4, 4), np.nan)
    return opt


def save_object_pose_table(opt, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    df = pd.DataFrame.from_records(opt)
    df['pose'] = df['pose'].apply(np.ndarray.tolist)
    df.to_csv(file_path, index=False)


def load_all_opts(scene_path, opt_file_name, convert2origin=False):
    extrinsics = load_extrinsics(f'{scene_path}/extrinsics.yml', to_mm=True)
    opt_all = []
    for camera_name in get_camera_names(scene_path):
        opt_path = f"{scene_path}/{camera_name}/{opt_file_name}"
        opt = load_object_pose_table(opt_path, only_valid_pose=True)
        if convert2origin:
            origin_camera = extrinsics[camera_name]
            opt['pose'] = [origin_camera @ p for p in opt['pose']]
        opt_all.append(opt)
    opt_all = np.hstack(opt_all)
    return opt_all


def load_primitives_table(file_path):
    df = pd.read_csv(file_path)
    pt = df.to_records(index=False)
    return pt


if __name__ == '__main__':
    # scene_path = '/home/gdk/data/1654267227_formated'
    # camera_names = get_camera_names(scene_path)
    # print(camera_names)

    # frame_number = frame_number(scene_path)
    # print(frame_number)

    # imgs_paths = '/home/gdk/data/1660754262/827312071794/foreground/_posecnn_results'
    # imgs_to_video(imgs_paths, imgs_paths, '.png_render.jpg')

    # test load extrinsics
    # ext_path = '/home/gdk/data/scene_220603104027/extrinsics.yml'
    # load_extrinsics(ext_path)

    # imgs_path = '/home/gdk/Repositories/VISOR-HOS/outputs/hos_postprocess'
    # imgs = load_images(imgs_path, 'jpg')
    # save_mp4(imgs, imgs_path + '/video.mp4', 30)

    # scene_name = 'scene_2211191439_ext'
    # cameras_intics = load_cameras_intrisics(scene_name)
    # print(cameras_intics)

    pt = load_primtives_table('/home/gdk/Data/kitchen_countertops/scene_2210232307_01/primitives.csv')
    print()