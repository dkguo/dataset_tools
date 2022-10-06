import os

import cv2
import yaml
import numpy as np

from modules.extrinsics.apriltag_detection import detect_april_tag
from modules.helper_functions.loaders import find_camera_seq


tag_size = 0.1
master_tag_id = 0

# TODO
def find_extrinsics(imagepath, camera_params, master_tag_id, visualize=False, save_path=None, verbose=False):
    poses, _ = detect_april_tag(cv2.imread(imagepath), camera_params, tag_size, visualize, save_path, verbose)

    for (id, pose) in poses:
        if id == master_tag_id:
            return np.linalg.inv(pose)

    return None


def process_extrinsics(demo_root):
    camera_seq = find_camera_seq(demo_root)

    extrinsics = {}
    for camera in camera_seq:
        imagepath = os.path.join(f'{demo_root}/{camera}', '000000_color.png')
        meta_path = os.path.join(f'{demo_root}/{camera}', 'meta.yml')
        with open(meta_path, "r") as stream:
            intrinsics = np.array(yaml.safe_load(stream)['INTRINSICS']).reshape(3, 3)
        camera_params = (intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])

        e = find_extrinsics(imagepath, camera_params, master_tag_id, visualize=True, save_path=None, verbose=False)
        extrinsics[camera] = e.tolist()

    #save
    save_dic = {'camera_seq': camera_seq,
                'master_tag_id': master_tag_id,
                'extrinsics': extrinsics}
    with open(f'{demo_root}/extrinsics.yml', 'w') as file:
        _ = yaml.dump(save_dic, file)


if __name__ == '__main__':
    demo_root = '/home/gdk/data/1660754262'
    process_extrinsics(demo_root)
