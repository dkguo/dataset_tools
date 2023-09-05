import glob
from multiprocessing import Pool

import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import *
from tqdm import tqdm

from dataset_tools.config import dataset_path
from dataset_tools.utils import get_camera_names


def collage_imgs(ims, num_rows=2):
    """
    num_rows: only works when len(imgs) >=7
    """
    imgs = ims.copy()
    if 7 <= len(imgs) <= 9:
        if num_rows == 2:
            for i in range(len(imgs), 8):
                imgs.append(np.zeros_like(imgs[0]))
        elif num_rows == 3:
            for i in range(len(imgs), 9):
                imgs.append(np.zeros_like(imgs[0]))

    if len(imgs) == 2:
        img = np.concatenate(imgs, axis=1)
    elif len(imgs) == 3:
        img = np.concatenate(imgs, axis=1)
    elif len(imgs) == 4:
        h_im_1 = np.concatenate([imgs[0], imgs[1]], axis=1)
        h_im_2 = np.concatenate([imgs[2], imgs[3]], axis=1)
        img = np.concatenate([h_im_1, h_im_2], axis=0)
    elif len(imgs) == 6:
        h_ims = []
        for i in range(2):
            h_ims.append(np.concatenate([imgs[3 * i], imgs[3 * i + 1], imgs[3 * i + 2]], axis=1))
        img = np.concatenate(h_ims, axis=0)
    elif len(imgs) == 8:
        h_im_1 = np.concatenate([imgs[0], imgs[1], imgs[2], imgs[3]], axis=1)
        h_im_2 = np.concatenate([imgs[4], imgs[5], imgs[6], imgs[7]], axis=1)
        img = np.concatenate([h_im_1, h_im_2], axis=0)
    elif len(imgs) == 9:
        h_ims = []
        for i in range(3):
            h_ims.append(np.concatenate([imgs[3 * i], imgs[3 * i + 1], imgs[3 * i + 2]], axis=1))
        img = np.concatenate(h_ims, axis=0)
    elif len(imgs) == 12:
        h_im_1 = collage_imgs(imgs[:6])
        h_im_2 = collage_imgs(imgs[6:])
        img = np.concatenate([h_im_1, h_im_2], axis=0)
    elif len(imgs) == 16:
        h_im_1 = collage_imgs(imgs[:8])
        h_im_2 = collage_imgs(imgs[8:])
        img = np.concatenate([h_im_1, h_im_2], axis=0)
    else:
        raise TypeError(f'{len(imgs)} images is not supported')
    return img


def add_border(img, color=1, width=10):
    """
    color: 0, 1, 2 -> r, g, b
    """
    border = np.zeros_like(img)
    border[:, :, color] = 255
    border[width:-width, width:-width, :] = img[width:-width, width:-width, :]
    return border


def add_texts(img, text_list, start_xy=(30, 30), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    text_list: [(text, color)], color=(0, 255, 0)
    """
    x, y = start_xy
    for i, (text, color) in enumerate(text_list):
        img = cv2.putText(img, text, (x, y + i * 30), font, 1, color, thickness=thickness)
    return img


def add_green_texts(img, text_list, start_xy=(30, 30), thickness=2, font_scale=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    text_list: [(text, color)], color=(0, 255, 0)
    """
    x, y = start_xy
    for i, text in enumerate(text_list):
        img = cv2.putText(img, text, (x, y + i * 30), font, font_scale, (0, 255, 0), thickness=thickness)
    return img


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


if __name__ == '__main__':
    add_frame_num_to_video('/home/gdk/Data/kitchen_countertops/scene_2210232307_01/video.mp4')

    scene_name = 'scene_230310200800'

    scene_names = ['scene_230313172946',
                   'scene_230313173036',
                   'scene_230313173113']

    for scene_name in scene_names:
        scene_path = f'{dataset_path}/{scene_name}'

        camera_paths = []
        for camera_name in get_camera_names(scene_path):
            camera_paths.append(f'{scene_path}/{camera_name}/rgb')

        with Pool() as pool:
            pool.map(png2video, camera_paths)

        video_paths = sorted(glob.glob(f'{scene_path}/camera_*/rgb/video.mp4'))
        save_path = f'{scene_path}/video.mp4'
        combine_videos(video_paths, save_path)

        add_frame_num_to_video(save_path)
