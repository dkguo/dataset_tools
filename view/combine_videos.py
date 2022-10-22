import glob
import os
import numpy as np
from moviepy.editor import *

from config import dataset_path
from view import scene_name


def combine_videos(video_paths, save_path):
    clips = []
    for video_path in video_paths:
        clips.append(VideoFileClip(video_path))

    final_clip = clips_array(np.reshape(clips, (2, -1)))

    final_clip = final_clip.fx(vfx.speedx, 1)

    final_clip.write_videofile(save_path)



if __name__ == '__main__':
    video_paths = sorted(glob.glob(f'{dataset_path}/{scene_name}/camera_*/video.mp4'))
    print(video_paths)

    save_path = f'{dataset_path}/{scene_name}/video.mp4'
    combine_videos(video_paths, save_path)
