import cv2
import yaml
import numpy as np

from loaders import get_camera_names, load_intrinsics
from process.apriltag_detection import detect_april_tag


def find_extrinsics(imagepath, camera_params, tag_size, master_tag_id, visualize=False, save_path=None, verbose=False):
    poses, _ = detect_april_tag(cv2.imread(imagepath), camera_params, tag_size, visualize, save_path, verbose)

    for (id, pose) in poses:
        if id == master_tag_id:
            return np.linalg.inv(pose)

    return None


def process_extrinsics(scene_path, tag_size, master_tag_id, master_image_id):
    camera_names = get_camera_names(scene_path)

    extrinsics = {}
    for camera in camera_names:
        image_path = f'{scene_path}/{camera}/rgb/{master_image_id:06}.png'
        intrinsics = load_intrinsics(f'{scene_path}/{camera}/camera_meta.yml')
        camera_params = (intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])

        e = find_extrinsics(image_path, camera_params, tag_size, master_tag_id, visualize=True)
        extrinsics[camera] = e.tolist()

    save_dic = {'tag_size': tag_size,
                'master_tag_id': master_tag_id,
                'master_image_id': master_image_id,
                'extrinsics': extrinsics}
    with open(f'{scene_path}/extrinsics.yml', 'w') as file:
        yaml.dump(save_dic, file)


if __name__ == '__main__':
    scene_path = '/Users/gdk/Downloads/data/scene_221012114441'
    process_extrinsics(scene_path, tag_size=0.06, master_tag_id=0, master_image_id=0)
