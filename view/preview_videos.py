import glob
from multiprocessing import Pool

import cv2
import numpy as np
from moviepy.editor import *
from tqdm import tqdm

from dataset_tools.config import dataset_path
from dataset_tools.loaders import get_camera_names
from dataset_tools.view.helpers import add_frame_num_to_video


def png2video(camera_path, frame_rate=30):
    print('Saving', camera_path)

    img = cv2.imread(f'{camera_path}/rgb/000000.png')
    dim = (img.shape[1], img.shape[0])
    out_video = cv2.VideoWriter(f'{camera_path}/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, dim)

    for img_path in tqdm(sorted(glob.glob(f'{camera_path}/rgb/*.png'))):
        img = cv2.imread(img_path)
        out_video.write(img)

    out_video.release()


def combine_videos(video_paths, save_path, speed=1):
    clips = []
    for video_path in video_paths:
        print('Combining', video_path)
        clips.append(VideoFileClip(video_path))

    final_clip = clips_array(np.reshape(clips, (2, -1)))
    final_clip = final_clip.fx(vfx.speedx, speed)
    final_clip.write_videofile(save_path)


if __name__ == '__main__':
    scene_name = 'scene_2211192313'

    scene_path = f'{dataset_path}/{scene_name}'

    camera_paths = []
    for camera_name in get_camera_names(scene_path):
        camera_paths.append(f'{scene_path}/{camera_name}')

    with Pool() as pool:
        pool.map(png2video, camera_paths)

    video_paths = sorted(glob.glob(f'{scene_path}/camera_*/video.mp4'))
    save_path = f'{scene_path}/video.mp4'
    combine_videos(video_paths, save_path)

    add_frame_num_to_video(save_path)
