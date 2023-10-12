import glob
import os

from dataset_tools.config import dataset_path


def get_scene_path(scene_name_path):
    if '/' in scene_name_path:
        return scene_name_path
    return f'{dataset_path}/{scene_name_path}'


def get_camera_names(scene_name_path):
    scene_path = get_scene_path(scene_name_path)
    folders = os.listdir(scene_path)
    camera_seq = []
    for folder in folders:
        if folder[:7] == 'camera_' and os.path.isdir(f'{scene_path}/{folder}'):
            camera_seq.append(folder)
    return sorted(camera_seq)


def get_num_frame(scene_name_path):
    scene_path = get_scene_path(scene_name_path)
    camera_names = get_camera_names(scene_path)
    nums = []
    for camera_name in camera_names:
        files = glob.glob(f"{scene_path}/{camera_name}/rgb/*.png")
        nums.append(len(files))
    return min(nums)


def get_available_frames(scene_name_path):
    scene_path = get_scene_path(scene_name_path)
    camera_names = get_camera_names(scene_path)
    frames = []
    for camera_name in camera_names:
        files = glob.glob(f"{scene_path}/{camera_name}/rgb/*.png")
        frames.append([int(f[-10:-4]) for f in files])
    frames = list(set.intersection(*map(set, frames)))
    return frames


def get_newest_scene_names(num=1):
    folders = os.listdir(dataset_path)
    scene_names = []
    for folder in folders:
        if folder[:6] == 'scene_' and os.path.isdir(f'{dataset_path}/{folder}'):
            scene_names.append(folder)
    scene_names.sort(reverse=True)
    return scene_names[:num]


def get_newest_scene_name():
    return get_newest_scene_names(1)[0]


def get_scene_name_from_path(path):
    for name in path.split('/'):
        if name[:6] == 'scene_':
            return name


def get_available_object_names(scene_name):
    file_names = os.listdir(f'{dataset_path}/{scene_name}/models')
    object_names = []
    for file_name in file_names:
        if file_name != 'plot' and file_name != 'kitchen' and 'csv' not in file_name:
            object_names.append(file_name)
    return object_names
