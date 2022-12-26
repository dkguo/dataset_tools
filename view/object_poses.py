import glob
import os

import cv2
import numpy as np
from tqdm import tqdm

from dataset_tools.config import dataset_path
from dataset_tools.loaders import get_camera_names, load_intrinsics, load_extrinsics, get_num_frame
from dataset_tools.view.renderer import create_scene, render_obj_pose, overlay_imgs
from dataset_tools.view.preview_videos import combine_videos


def view_obj_poses(scene_name, camera_ids, obj_poses, save_folder_name):
    """
    Args:
        camera_ids: list or 'all'
        obj_poses: dict[obj_id] = poses, pose must be numpy array
        save_path: must end with mp4
    """
    # assert save_path[-4:] == '.mp4', 'must save to a file ends with .mp4'
    # save_dir = os.path.basename(save_path)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    scene_path = f'{dataset_path}/{scene_name}'
    camera_names = get_camera_names(scene_path)
    extrinsics = load_extrinsics(f'{scene_path}/cameras_meta.yml')
    num_frames = get_num_frame(scene_path)

    frames_combined_obj_T = {}
    for obj_id, poses in obj_poses.items():
        for frame_i, pose in enumerate(poses):
            if frame_i not in frames_combined_obj_T:
                frames_combined_obj_T[frame_i] = {}
            if not np.isnan(pose).any():
                frames_combined_obj_T[frame_i][obj_id] = pose

    # Convert back to each camera
    for camera_name in camera_names:
        output_dir = f'{scene_path}/{camera_name}/object_pose/{save_folder_name}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        print(f'{output_dir}/video.mp4')
        out_video = cv2.VideoWriter(f'{output_dir}/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

        intrinsics = load_intrinsics(f'{scene_path}/{camera_name}/camera_meta.yml')
        py_renderer = create_scene(intrinsics)

        for frame in tqdm(range(num_frames)):
            im_file = f'{scene_path}/{camera_name}/rgb/{frame:06d}.png'
            im_ori = cv2.imread(im_file)

            if len(frames_combined_obj_T[frame]) == 0:
                out_im = im_ori
            else:
                dict_id_poses = {}

                for id, origin_object in frames_combined_obj_T[frame].items():
                    id = int(id)
                    camera_origin = np.linalg.inv(extrinsics[camera_name])
                    pose = camera_origin @ origin_object
                    if id in dict_id_poses:
                        dict_id_poses[id].append(pose)
                    else:
                        dict_id_poses[id] = [pose]

                rendered_im = render_obj_pose(py_renderer, dict_id_poses)
                out_im = overlay_imgs(im_ori, rendered_im)

            cv2.imwrite(f'{output_dir}/{frame:06d}.png', out_im)
            out_video.write(out_im)

        out_video.release()

    video_paths = sorted(glob.glob(f'{scene_path}/camera_*/object_pose/{save_folder_name}/video.mp4'))
    save_path = f'{scene_path}/object_pose/{save_folder_name}/video.mp4'
    combine_videos(video_paths, save_path)