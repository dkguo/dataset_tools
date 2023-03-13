import cv2
import numpy as np
import yaml
from tqdm import tqdm

from dataset_tools.config import dataset_path
from dataset_tools.loaders import load_cameras_intrisics, load_cameras_extrinsics, get_num_frame, \
    load_imgs_across_cameras, get_camera_names, intr2param
from dataset_tools.record.apriltag_detection import verify_calibration, detect_april_tag, draw_pose
from dataset_tools.view.helpers import collage_imgs


def extract_infra_pose(scene_name, infra_tag_id=1, infra_tag_size=0.06):
    scene_path = f'{dataset_path}/{scene_name}'

    cameras_intr = load_cameras_intrisics(scene_name)
    cameras_ext = load_cameras_extrinsics(scene_name)

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
    infra_pose = {'SINK_UNIT_ORIGIN': (cameras_ext[camera] @ pose).tolist(),
                  'TAG_SIZE': 0.06,
                  'TAG_ID': 1}

    with open(f'{scene_path}/infra_pose.yml', 'w') as file:
        yaml.dump(infra_pose, file)


def load_infra_pose(scene_name):
    with open(f'{dataset_path}/{scene_name}/infra_pose.yml') as file:
        infra_pose = np.array(yaml.safe_load(file)['SINK_UNIT_ORIGIN']).reshape(4, 4)
    return infra_pose


def view_april_tag_pose(tag_pose, imgs, cameras_intr, camera_ext, tag_size):
    for img, intr, ext in zip(imgs, cameras_intr.values(), camera_ext.values()):
        pose = np.linalg.inv(ext) @ tag_pose

        camera_params = intr2param(intr)
        draw_pose(img, camera_params, tag_size, pose)

    preview = collage_imgs(imgs)
    cv2.imshow('verify', preview)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()


def view_infra_pose(scene_name):
    scene_path = f'{dataset_path}/{scene_name}'
    imgs = load_imgs_across_cameras(scene_path, get_camera_names(scene_path), f'{0:06d}.png')
    cameras_intr = load_cameras_intrisics(scene_name)
    cameras_ext = load_cameras_extrinsics(scene_name)
    view_april_tag_pose(infra_pose, imgs, cameras_intr, cameras_ext, tag_size)


if __name__ == '__main__':
    scene_name = 'scene_2303102008'

    tag_size = 0.06

    # extract_infra_pose(scene_name)
    infra_pose = load_infra_pose(scene_name)
    view_infra_pose(scene_name)

