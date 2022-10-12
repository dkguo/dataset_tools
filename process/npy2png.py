import os.path
import shutil
import glob
import numpy as np

from bop_toolkit.bop_toolkit_lib.inout import save_im, save_depth
from loaders import get_camera_names, get_depth_scale


def npy2png(scene_path, start_frame=0):
    camera_names = get_camera_names(scene_path)
    for camera in camera_names:
        print(f'converting npy2png for {os.path.basename(scene_path)} {camera}...')
        camera_path = f'{scene_path}/{camera}'

        if os.path.exists(f'{camera_path}/rgb'):
            shutil.rmtree(f'{camera_path}/rgb')
        os.mkdir(f'{camera_path}/rgb')
        if os.path.exists(f'{camera_path}/depth'):
            shutil.rmtree(f'{camera_path}/depth')
        os.mkdir(f'{camera_path}/depth')

        for i, img_path in enumerate(sorted(glob.glob(f'{camera_path}/rgb_npy/*.npy'))[start_frame:]):
            save_path = f'{camera_path}/rgb/{i:06d}.png'
            img = np.load(img_path)
            save_im(save_path, img[:, :, ::-1])

        for i, img_path in enumerate(sorted(glob.glob(f'{camera_path}/depth_npy/*.npy'))[start_frame:]):
            save_path = f'{camera_path}/depth/{i:06d}.png'
            img = np.load(img_path)
            depth_scale = get_depth_scale(f'{camera_path}/camera_meta.yml')
            img = img * depth_scale
            save_depth(save_path, img)

        open(f'{camera_path}/rgb_npy/start_frame={start_frame}', 'w')


if __name__ == '__main__':
    scene_path = '/Users/gdk/Downloads/data/scene_221012114441'
    npy2png(scene_path, start_frame=40)
