import glob
import os
import shutil

import numpy as np
import yaml

from dataset_tools.config import dataset_path
from dataset_tools.utils.name import get_camera_names


def copy_extrinsics(scene_name):
    ext_path = sorted(glob.glob(f'{dataset_path}/scene_*_ext'), reverse=True)[0] + '/extrinsics.yml'
    shutil.copy(ext_path, f'{dataset_path}/{scene_name}/extrinsics.yml')
    print(f'Copied extrinsics from {ext_path} to {dataset_path}/{scene_name}/extrinsics.yml')


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
    if os.path.exists(f'{dataset_path}/{scene_name}/cameras_meta.yml'):
        cameras_meta = load_cameras_meta(scene_name)
        cameras_intics = cameras_meta['INTRINSICS']
        for camera_name, intri in cameras_intics.items():
            cameras_intics[camera_name] = np.array(intri)
        return cameras_intics
    else:
        cameras_intrinsics = {}
        camera_names = get_camera_names(f'{dataset_path}/{scene_name}')
        for camera_name in camera_names:
            cameras_intrinsics[camera_name] = load_intrinsics(
                f'{dataset_path}/{scene_name}/{camera_name}/camera_meta.yml')
        return cameras_intrinsics


def load_cameras_extrinsics(scene_name):
    if os.path.exists(f'{dataset_path}/{scene_name}/cameras_meta.yml'):
        return load_extrinsics(f'{dataset_path}/{scene_name}/cameras_meta.yml')
    else:
        return load_extrinsics(f'{dataset_path}/{scene_name}/extrinsics.yml')


def get_depth_scale(camera_meta_path, convert2unit='mm'):
    with open(camera_meta_path, "r") as stream:
        depth_unit = yaml.safe_load(stream)['DEPTH_UNIT']
    dict_unit_scale = {'m': 1000.0, 'mm': 1.0, 'μm': 0.1}
    depth_scale = dict_unit_scale[depth_unit] / dict_unit_scale[convert2unit]
    return depth_scale