import glob

import numpy as np
from moviepy.editor import *

from config import dataset_path
from loaders import get_camera_names, load_images, save_mp4


def png2video(scene_path, frame_rate=30):
    for camera_name in get_camera_names(scene_path):
        camera_path = f'{scene_path}/{camera_name}'
        print('Loading', camera_path)
        imgs = load_images(f'{camera_path}/rgb', '.png')
        save_mp4(imgs, f'{camera_path}/video.mp4', frame_rate=frame_rate)


def combine_videos(video_paths, save_path, speed=1):
    clips = []
    for video_path in video_paths:
        print('Combining', video_path)
        clips.append(VideoFileClip(video_path))

    final_clip = clips_array(np.reshape(clips, (2, -1)))
    final_clip = final_clip.fx(vfx.speedx, speed)
    final_clip.write_videofile(save_path)


if __name__ == '__main__':
    scene_name = 'scene_2210232307_01'

    scene_path = f'{dataset_path}/{scene_name}'

    png2video(scene_path)

    video_paths = sorted(glob.glob(f'{scene_path}/camera_*/video.mp4'))
    save_path = f'{scene_path}/video.mp4'
    combine_videos(video_paths, save_path)