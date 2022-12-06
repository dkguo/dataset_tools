import csv
import json
import os
import glob
import re

import cv2
import numpy as np
import yaml
import scipy.io


from dataset_tools.dataset_config import dataset_path


def get_camera_names(scene_path):
    folders = os.listdir(scene_path)
    camera_seq = []
    for folder in folders:
        if folder[:7] == 'camera_' and os.path.isdir(f'{scene_path}/{folder}'):
            camera_seq.append(folder)
    return sorted(camera_seq)


def frame_number(scene_path):
    camera_seq = get_camera_names(scene_path)
    nums = []
    for camera in camera_seq:
        files = glob.glob(f'{scene_path}/{camera}/rgb/*.png')
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


def load_extrinsics(extrinsics_path):
    with open(extrinsics_path, 'r') as file:
        ext = yaml.full_load(file)
    if 'EXTRINSICS' in ext:
        ext = ext['EXTRINSICS']
    for k, e in ext.items():
        ext[k] = np.array(e)
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


def load_ground_truth(gt_path):
    """
    :return: dict_imid_objid_poses
    """
    with open(gt_path) as f:
        im_gts = json.load(f)

    gt_dict_imid_objid_poses = {}
    for im_id, gts in im_gts.items():
        gt_dict_id_poses = {}
        for gt in gts:
            R_gt = np.array(gt['cam_R_m2c']).reshape((3, 3))
            t_gt = np.array(gt['cam_t_m2c'])  # mm
            p_gt = np.c_[R_gt, t_gt]

            obj_id = gt['obj_id']
            if obj_id not in gt_dict_id_poses:
                gt_dict_id_poses[obj_id] = []
            gt_dict_id_poses[obj_id].append(p_gt)
        gt_dict_imid_objid_poses[int(im_id)] = gt_dict_id_poses
    return gt_dict_imid_objid_poses


def load_bop_est_pose(bop_results_path, scene_id, image_id):
    est_dict_id_poses = {}
    with open(bop_results_path) as f:
        reader = csv.reader(f)
        for row in reader:
            row_scene_id, im_id, obj_id, score, R, t, time = row
            if row_scene_id == str(scene_id) and im_id == str(image_id):
                R = np.fromstring(R, sep=' ').reshape((3, 3))
                t = np.fromstring(t, sep=' ')   # mm
                pose = np.c_[R, t]
                if int(obj_id) not in est_dict_id_poses:
                    est_dict_id_poses[int(obj_id)] = [pose]
                else:
                    est_dict_id_poses[int(obj_id)].append(pose)
    return est_dict_id_poses


def load_object_poses(file_path):
    """
    This function loads object poses assuming there is only one identical object in each frame.
    :param file_path: the file has dict[frame][object_id] = pose

    :return: object_poses: dict[object_id] = pose
    """
    with open(file_path) as f:
        frame_object_poses = json.load(f)

    # find obj keys
    obj_ids = set()
    for object_poses in frame_object_poses.values():
        for obj_id in object_poses.keys():
            obj_ids.add(obj_id)

    object_poses = {}
    for obj_id in obj_ids:
        poses = []
        last_frame = int(list(frame_object_poses.keys())[-1])
        for frame in range(last_frame + 1):
            if str(frame) in frame_object_poses:
                if obj_id in frame_object_poses[str(frame)]:
                    pose = np.array(frame_object_poses[str(frame)][obj_id])
                    assert pose.shape == (4, 4)
                    poses.append(np.array(frame_object_poses[str(frame)][obj_id]))
                else:
                    poses.append(None)
            else:
                poses.append(None)
        object_poses[obj_id] = poses

    return object_poses


if __name__ == '__main__':
    # scene_path = '/home/gdk/data/1654267227_formated'
    # camera_seq = get_camera_names(scene_path)
    # print(camera_seq)

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

    scene_name = 'scene_2211191439_ext'
    cameras_intics = load_cameras_intrisics(scene_name)
    print(cameras_intics)
