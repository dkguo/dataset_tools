import glob
import json
import os.path

from loaders import get_camera_names, load_intrinsics


def link_to_bop_format(scene_path):
    for i, camera in enumerate(get_camera_names(scene_path)):
        scene_id = os.path.basename(scene_path).split('_')[1]
        bop_path = f'{os.path.dirname(scene_path)}/bop_format/val/{scene_id}{i:02d}'
        os.symlink(f'{scene_path}/{camera}', bop_path)

        # create scene_camera_json
        scene_camera = {}
        intrinsics = load_intrinsics(f'{bop_path}/camera_meta.yml')
        for k in range(len(glob.glob(f'{bop_path}/rgb/*.png'))):
            scene_camera[k] = {'cam_K': intrinsics.reshape([-1]).tolist(), 'depth_scale': 1.0}
        with open(f'{bop_path}/scene_camera.json', 'w') as f:
            json.dump(scene_camera, f)


if __name__ == '__main__':
    scene_path = '/home/gdk/data/scene_220603104027'
    link_to_bop_format(scene_path)
