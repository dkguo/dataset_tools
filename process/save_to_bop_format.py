import glob
import json
import os.path

from modules.helpers.loaders import find_camera_seq, load_intrinsics


def record2bop(demo_path, first_camera_save_path):
    for i, camera in enumerate(find_camera_seq(demo_path)):
        save_path = first_camera_save_path[:-1] + str(i)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            os.mkdir(f'{save_path}/rgb')
            os.mkdir(f'{save_path}/depth')

        # link color images
        for k, img_name in enumerate(sorted(glob.glob(f'{demo_path}/{camera}/*color.png'))):
            dst_path = f'{save_path}/rgb/{k:06d}.png'
            if not os.path.exists(dst_path):
                os.symlink(img_name, dst_path)

        # link depth images
        for k, img_name in enumerate(sorted(glob.glob(f'{demo_path}/{camera}/*depth.png'))):
            dst_path = f'{save_path}/depth/{k:06d}.png'
            if not os.path.exists(dst_path):
                os.symlink(img_name, dst_path)

        # create scene_camera_json
        s = {}
        K = load_intrinsics(f'{demo_path}/{camera}/meta.yml')
        for k, img_name in enumerate(sorted(glob.glob(f'{demo_path}/{camera}/*color.png'))):
            s[k] = {'cam_K': K.reshape([-1]).tolist(), 'depth_scale': 1.0}
        with open(f'{save_path}/scene_camera.json', 'w') as f:
            json.dump(s, f)


if __name__ == '__main__':
    demo_path = '/home/gdk/data/1660754262'
    first_camera_save_path = '/home/gdk/data/bop_format/val/000000'
    record2bop(demo_path, first_camera_save_path)