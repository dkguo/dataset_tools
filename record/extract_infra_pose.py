import csv
from pprint import pprint

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from dataset_tools.config import dataset_path
from dataset_tools.loaders import load_cameras_intrisics, load_cameras_extrinsics, get_num_frame, \
    load_imgs_across_cameras, get_camera_names, intr2param, load_object_pose_table
from dataset_tools.record.apriltag_detection import verify_calibration, detect_april_tag, draw_pose, draw_pose_axes
from dataset_tools.view.helpers import collage_imgs
from dataset_tools.view.renderer import create_renderer, render_obj_pose, overlay_imgs, set_intrinsics


def extract_infra_pose(scene_name, infra_tag_id=1, infra_tag_size=0.06):
    scene_path = f'{dataset_path}/{scene_name}'

    cameras_intr = load_cameras_intrisics(scene_name)
    cameras_ext = load_cameras_extrinsics(scene_name)

    print('Extracting infra pose')
    min_ef = np.Inf
    for frame in tqdm(range(10)):
        imgs = load_imgs_across_cameras(scene_path, get_camera_names(scene_path), f'{frame:06d}.png')
        for camera, img in zip(get_camera_names(scene_path), imgs):
            intr = cameras_intr[camera]
            param = intr2param(intr)
            poses, overlay = detect_april_tag(img, param, infra_tag_size)
            for tag_id, pose, ef in poses:
                if tag_id == infra_tag_id and ef < min_ef:
                    min_ef = ef
                    best_pose = (camera, pose)

    camera, pose = best_pose

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

    tag_pose = cameras_ext[camera] @ tag_pose

    f = open(f'{scene_path}/infra_poses.csv', 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['obj_id', 'name', 'frame', 'pose'])
    csv_writer.writerow([100, 'sink_unit', 0, tag_pose.tolist()])

    rx = np.array([[1, 0, 0, 0],
                   [0, 0, -1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]])
    ry = np.array([[0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [-1, 0, 0, 0],
                   [0, 0, 0, 1]])
    rz = np.array([[0, -1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    t = np.array([[0, 0, 1, 0.47793],
                  [1, 0, 0, -0.06707],
                  [0, 1, 0, -0.29],
                  [0, 0, 0, 1]])
    # r = np.array([[0, -1, 0, 0],
    #               [0, 0, -1, 0],
    #               [1, 0, 0, 0],
    #               [0, 0, 0, 1]])
    # sink_only_pose = tag_pose @ rx @ ry
    sink_only_pose = tag_pose @ t
    # sink_only_pose[:3, 3] += np.array([0.06707, 0.47793, 0.29])
    # sink_only_pose[:3, 3] -= np.array([0.47793, -0.06707, -0.29])
    csv_writer.writerow([101, 'sink_only', 0, sink_only_pose.tolist()])


def load_infra_pose(scene_name):
    ipt = load_object_pose_table(f"{dataset_path}/{scene_name}/infra_poses.csv", only_valid_pose=True)
    infra_pose = ipt[ipt['obj_id'] == 100]['pose'][0]
    return infra_pose


def view_april_tag_pose(tag_pose, imgs, cameras_intr, camera_ext, tag_size):
    # original tags
    overlays = []
    for img, intr in zip(imgs, cameras_intr.values()):
        poses, overlay = detect_april_tag(img, intr2param(intr), tag_size)
        overlays.append(overlay)
    overlay = collage_imgs(overlays)
    cv2.imshow('orignal tags', overlay)

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

    preview = collage_imgs(imgs)
    cv2.imshow('verify', preview)

    preview = collage_imgs(re_ims)
    cv2.imshow('verify 2', preview)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()


def view_infra_pose(scene_name):
    scene_path = f'{dataset_path}/{scene_name}'
    imgs = load_imgs_across_cameras(scene_path, get_camera_names(scene_path), f'{0:06d}.png')
    cameras_intr = load_cameras_intrisics(scene_name)
    cameras_ext = load_cameras_extrinsics(scene_name)
    infra_pose = load_infra_pose(scene_name)
    view_april_tag_pose(infra_pose, imgs, cameras_intr, cameras_ext, tag_size=0.06)


if __name__ == '__main__':
    # scene_names = ['scene_230310200800']

    scene_names = ['scene_230313171600',
                   'scene_230313171700',
                   'scene_230313171800',
                   'scene_230313171900',
                   'scene_230313172000',
                   'scene_230313172100',
                   'scene_230313172200',
                   'scene_230313172537',
                   'scene_230313172613',
                   'scene_230313172659',
                   'scene_230313172735',
                   'scene_230313172808',
                   'scene_230313172840',
                   'scene_230313172915',
                   'scene_230313172946',
                   'scene_230313173036',
                   'scene_230313173113']

    for scene_name in scene_names:
        extract_infra_pose(scene_name)

    # view_infra_pose(scene_name)

