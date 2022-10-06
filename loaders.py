import csv
import json
import os
import glob
import re

import cv2
import numpy as np
import yaml


def find_camera_seq(demo_root):
    folders = os.listdir(demo_root)
    camera_seq = []
    for f in folders:
        if f[:5].isnumeric():
            camera_seq.append(f)
    return camera_seq


def frame_number(demo_root):
    camera_seq = find_camera_seq(demo_root)
    nums = []
    for camera in camera_seq:
        files = glob.glob(f'{demo_root}/{camera}/*depth.png')
        nums.append(len(files))
    return min(nums)


def save_video(imgs, save_path, video_name='video'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print(save_path, f'Saving {video_name}.mp4')
    out_video = cv2.VideoWriter(f'{save_path}/{video_name}.mp4',
                                cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
    for im in imgs:
        out_video.write(im)
    out_video.release()


def save_imgs(imgs, save_path, uniqname=None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print(save_path, f'Saving {uniqname if uniqname is not None else ""}.png')
    for frame, im in enumerate(imgs):
        cv2.imwrite(f'{save_path}/{frame}{uniqname if uniqname is not None else ""}.png', im)


def imgs_to_video(imgs_path, save_path, uniqname):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print(save_path, 'Converting imgs to video.mp4')
    out_video = cv2.VideoWriter(f'{save_path}/video.mp4',
                                cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))

    imgs = load_images(imgs_path, uniqname)
    for im in imgs:
        out_video.write(im)
    out_video.release()


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


def load_npys(demo_root, camera, img_type, verbose=False):
    npys = []
    num_frame = frame_number(demo_root)
    for frame in range(num_frame):
        path = f'{demo_root}/{camera}/{frame:06d}_{img_type}.npy'
        if verbose:
            print(path)
        npys.append(np.load(path))
    return npys


def load_intrinsics(intrinsics_path):
    with open(intrinsics_path, "r") as stream:
        intrinsics = np.array(yaml.safe_load(stream)['INTRINSICS']).reshape(3, 3)
    return intrinsics


def load_ground_truth(gt_path):
    with open(gt_path) as f:
        im_gts = json.load(f)

    gt_dict_iim_iobj_poses = {}
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
        gt_dict_iim_iobj_poses[int(im_id)] = gt_dict_id_poses
    return gt_dict_iim_iobj_poses


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


if __name__ == '__main__':
    # demo_root = '/home/gdk/Documents/data/1652826411'
    # camera_seq = find_camera_seq(demo_root)
    # print(camera_seq)
    # frame_number = frame_number(demo_root)
    # print(frame_number)

    imgs_paths = '/home/gdk/data/1660754262/827312071794/foreground/_posecnn_results'
    imgs_to_video(imgs_paths, imgs_paths, '.png_render.jpg')
