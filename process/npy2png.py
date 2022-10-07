import os.path

import cv2
import glob
import numpy as np

from bop_toolkit_lib.inout import save_im, save_depth
from loaders import get_camera_names, get_depth_scale


def npy2png(scene_path, start_frame=0):
    camera_names = get_camera_names(scene_path)
    for camera in camera_names:
        print(f'converting {camera} npy2png...')
        camera_path = f'{scene_path}/{camera}'
        if not os.path.exists(f'{camera_path}/rgb'):
            os.mkdir(f'{camera_path}/rgb')
        if not os.path.exists(f'{camera_path}/depth'):
            os.mkdir(f'{camera_path}/depth')

        for img_path in glob.glob(f'{camera_path}/rgb_npy/*.npy'):
            save_path = f'{camera_path}/rgb/{os.path.basename(img_path)[:-4]}.png'
            img = np.load(img_path)
            save_im(save_path, img[:, :, ::-1])

        for img_path in glob.glob(f'{camera_path}/depth_npy/*.npy'):
            save_path = f'{camera_path}/depth/{os.path.basename(img_path)[:-4]}.png'
            img = np.load(img_path)
            depth_scale = get_depth_scale(f'{camera_path}/camera_meta.yml')
            img = img * depth_scale
            save_depth(save_path, img)


if __name__ == '__main__':
    scene_path = '/home/gdk/data/scene_220603104027'
    npy2png(scene_path)
