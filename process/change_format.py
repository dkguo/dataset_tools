import shutil
from datetime import datetime
import os.path

import yaml

from loaders import get_camera_names, load_intrinsics


def change_old_to_new_data_format(old_scene_path):
    # 1. create new scene folder
    old_scene_name = os.path.basename(old_scene_path)
    r = old_scene_name.split('_')
    timestamp = int(r[0])
    dt_obj = datetime.fromtimestamp(timestamp)
    dt = dt_obj.strftime("%y%m%d%H%M%S")
    note = f'_{r[1]}' if len(r) > 1 else ''
    new_scene_name = f'scene_{dt}{note}'
    new_scene_path = f'{os.path.dirname(old_scene_path)}/{new_scene_name}'
    if not os.path.exists(new_scene_path):
        shutil.copytree(old_scene_path, new_scene_path)

    # 2. convert camera names
    folders = os.listdir(new_scene_path)
    old_camera_names = []
    for f in folders:
        if f[:5].isnumeric():
            old_camera_names.append(f)
    for i, camera_name in enumerate(old_camera_names):
        new_camera_name = f'camera_{i + 1:02d}_{camera_name}'
        os.rename(f'{new_scene_path}/{camera_name}', f'{new_scene_path}/{new_camera_name}')

    # 3. for each camera: move npy to folders, generate images, move others to results folder
    for camera_name in get_camera_names(new_scene_path):
        camera_path = f'{new_scene_path}/{camera_name}'
        results_path = f'{camera_path}/results'
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        rgbnpy_path = f'{camera_path}/rgb_npy'
        if not os.path.exists(rgbnpy_path):
            os.mkdir(rgbnpy_path)
        depthnpy_path = f'{camera_path}/depth_npy'
        if not os.path.exists(depthnpy_path):
            os.mkdir(depthnpy_path)
        for f in os.listdir(f'{camera_path}'):
            if '.' not in f and f != 'results' and 'rgb' not in f and 'depth' not in f:
                os.rename(f'{camera_path}/{f}', f'{results_path}/{f}')

            if '.png' in f:
                os.remove(f'{camera_path}/{f}')

            if 'color.npy' in f:
                new_name = f.replace('_color', '')
                os.rename(f'{camera_path}/{f}', f'{camera_path}/rgb_npy/{new_name}')

            if 'depth.npy' in f:
                new_name = f.replace('_depth', '')
                os.rename(f'{camera_path}/{f}', f'{camera_path}/depth_npy/{new_name}')

        intric = load_intrinsics(f'{camera_path}/meta.yml')
        with open(f'{camera_path}/camera_meta.yml', 'w') as file:
            assert True
            save_str = {'INTRINSICS': intric.reshape(-1).tolist(),
                        'DEPTH_UNIT': 'mm',
                        'FRAME_WIDTH': 640,
                        'FRAME_HEIGHT': 480}
            yaml.dump(save_str, file)
        os.remove(f'{camera_path}/meta.yml')


if __name__ == '__main__':
    paths = ['/Users/gdk/Downloads/data/1665589481']
    for old_path in paths:
        change_old_to_new_data_format(old_path)
