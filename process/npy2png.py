import cv2
import glob
import numpy as np

from bop_toolkit_lib.inout import save_im, save_depth
from modules.helpers.loaders import find_camera_seq


def npy2png(demo_root):
    camera_seq = find_camera_seq(demo_root)
    for camera in camera_seq:
        print(f'converting {camera}...')
        for img_name in glob.glob(f'{demo_root}/{camera}/*.npy'):
            save_path = f'{img_name[:-4]}.png'
            img = np.load(img_name)
            if 'color' in img_name:
                save_im(save_path, img[:, :, ::-1])
            elif 'depth' in img_name:
                img = img / 10.0
                save_depth(save_path, img)


if __name__ == '__main__':
    demo_root = '/home/gdk/data/1664878814-D405'
    npy2png(demo_root)
