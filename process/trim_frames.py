import glob
import os

from dataset_tools.dataset_config import dataset_path
from dataset_tools.loaders import get_camera_names

if __name__ == '__main__':
    scene_name = 'scene_2211192313'
    start_frame = 130
    end_frame = 650  # inclusive

    for camera_name in get_camera_names(f'{dataset_path}/{scene_name}'):
        camera_path = f'{dataset_path}/{scene_name}/{camera_name}'

        color_paths = sorted(glob.glob(f'{camera_path}/rgb/*.png'))
        depth_paths = sorted(glob.glob(f'{camera_path}/depth/*.png'))

        for cp, dp in zip(color_paths, depth_paths):
            frame_num = int(os.path.basename(cp)[:6])
            if frame_num < start_frame:
                os.remove(cp)
                os.remove(dp)
            elif frame_num <= end_frame:
                os.rename(cp, f'{camera_path}/rgb/{frame_num-start_frame:06d}.png')
                os.rename(dp, f'{camera_path}/depth/{frame_num-start_frame:06d}.png')
            else:
                os.remove(cp)
                os.remove(dp)
