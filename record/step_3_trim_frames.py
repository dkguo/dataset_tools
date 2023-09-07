import glob
import os

from dataset_tools.config import dataset_path
from dataset_tools.utils.name import get_camera_names, get_newest_scene_name

if __name__ == '__main__':
    # scene_name = 'scene_230825131826'
    scene_name = get_newest_scene_name()
    start_frame = 78
    end_frame = 256  # inclusive

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
