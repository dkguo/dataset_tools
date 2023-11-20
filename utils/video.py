import glob
from multiprocessing import Pool

import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import *
from tqdm import tqdm

from dataset_tools.config import dataset_path
from dataset_tools.utils.image import add_green_texts
from dataset_tools.utils.name import get_camera_names


def add_text_video(dict_frame_text, video_path, save_path, frame_rate=None,
                   start_xy=(50, 100), thickness=10, font_scale=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('video is not open')

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS) if frame_rate is None else frame_rate)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Add text to {video_path}')
    print(f'Saving at {save_path}')
    print(f'{frame_rate} fps, dim: {width}, {height}')

    if save_path == video_path:
        print(f'{video_path[:-4]}_tmp.mp4')
        out_video = cv2.VideoWriter(f'{video_path[:-4]}_tmp.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                    frame_rate, (width, height))
    else:
        out_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if i in dict_frame_text:
            frame = add_green_texts(frame, dict_frame_text[i], start_xy=start_xy, thickness=thickness,
                                    font_scale=font_scale)

        out_video.write(frame)
        cv2.imshow('f', frame)
        cv2.waitKey(1)
        i += 1

    out_video.release()

    if save_path == video_path:
        os.rename(f'{video_path[:-4]}_tmp.mp4', video_path)


def add_frame_num_to_video(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    dict_frame_text = {}
    for i in range(num_frames):
        dict_frame_text[i] = [f'Frame {i}']

    add_text_video(dict_frame_text, video_path, video_path, thickness=6)


def plot_poses_3d(poses):
    coods = []
    for pose in poses:
        if pose is not None:
            p_object = pose[:3, 3]
            R = pose[:3, :3]
            p_x = R @ [0.01, 0, 0]
            p_y = R @ [0, 0.01, 0]
            p_z = R @ [0, 0, 0.01]
            cood = np.r_[[np.r_[p_object, p_x],
                          np.r_[p_object, p_y],
                          np.r_[p_object, p_z]]]
            coods.append(cood)

    coods = np.array(coods)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, c in zip(range(3), ['b', 'r', 'g']):
        s = coods[:, i, :].reshape((-1, 6))
        X, Y, Z, U, V, W = zip(*s)
        ax.quiver(X, Y, Z, U, V, W, color=c)
    plt.show()


def png2video(imgs_dir, frame_rate=30):
    print(f'Saving {imgs_dir}/video.mp4')

    img = cv2.imread(f'{imgs_dir}/000000.png')
    dim = (img.shape[1], img.shape[0])
    out_video = cv2.VideoWriter(f'{imgs_dir}/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, dim)

    for img_path in tqdm(sorted(glob.glob(f'{imgs_dir}/*.png'))):
        img = cv2.imread(img_path)
        out_video.write(img)

    out_video.release()


def combine_videos(video_paths, save_path, speed=1, video_shape=(2, -1)):
    clips = []
    for video_path in video_paths:
        print('Combining', video_path)
        clips.append(VideoFileClip(video_path))

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    final_clip = clips_array(np.reshape(clips, video_shape))
    final_clip = final_clip.fx(vfx.speedx, speed)
    final_clip.write_videofile(save_path)


def save_mp4(imgs, video_save_path, frame_rate=15):
    assert video_save_path[-3:] == 'mp4', 'video_save_path has to end with mp4'
    dim = (imgs[0].shape[1], imgs[0].shape[0])
    print(f'Saving {video_save_path}')
    out_video = cv2.VideoWriter(f'{video_save_path}', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, dim)
    for im in imgs:
        out_video.write(im)
    out_video.release()


if __name__ == '__main__':
    # add_frame_num_to_video('/home/gdk/Data/kitchen_countertops/scene_2210232307_01/video.mp4')
    #
    # scene_name = 'scene_230310200800'
    #
    # scene_names = ['scene_230313172946',
    #                'scene_230313173036',
    #                'scene_230313173113']
    #
    # for scene_name in scene_names:
    #     scene_path = f'{dataset_path}/{scene_name}'
    #
    #     camera_paths = []
    #     for camera_name in get_camera_names(scene_path):
    #         camera_paths.append(f'{scene_path}/{camera_name}/rgb')
    #
    #     with Pool() as pool:
    #         pool.map(png2video, camera_paths)
    #
    #     video_paths = sorted(glob.glob(f'{scene_path}/camera_*/rgb/video.mp4'))
    #     save_path = f'{scene_path}/video.mp4'
    #     combine_videos(video_paths, save_path)
    #
    #     add_frame_num_to_video(save_path)

    img_dir = '/home/gdk/Data/kitchen_lab/scene_230911173348_blue_bowl/camera_01_827312072396/rgb'
    png2video(img_dir)
