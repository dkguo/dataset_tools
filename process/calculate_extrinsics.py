import cv2
import yaml
import numpy as np

from config import dataset_path
from loaders import get_camera_names, load_intrinsics
from process.apriltag_detection import detect_april_tag
from process.helpers import add_texts, add_border, collage_imgs

apriltag_detect_error_thres = 0.05


def process_extrinsics(camera_names_image_params, tag_size, save_path=None):
    """

    Args:
        camera_names_image_params: dict[camera_name] = (image, params), camera_01_* is used as master
        tag_size

    Returns:
        camera_extrinsics: dict[camera] = extrinsics
        preview_img
    """
    camera_names = sorted(camera_names_image_params.keys())

    correspondences = []
    overlays = []
    for camera_id, camera_name in enumerate(camera_names):
        image, camera_params = camera_names_image_params[camera_name]

        poses_errs, overlay = detect_april_tag(image, camera_params, tag_size)

        text_list = []
        for tag_id, pose, error in poses_errs:
            if error < apriltag_detect_error_thres:
                correspondences.append((tag_id, camera_id, pose))
                text_list.append((f'tag: {tag_id}, error: {error:.3f}', (0, 255, 0)))
            else:
                text_list.append((f'tag: {tag_id}, error: {error:.3f}', (0, 0, 255)))

        overlay = add_texts(overlay, text_list, start_xy=(30, 50), thickness=2)
        overlays.append(overlay)

    camera_extrinsics = {0: np.identity(4)}
    tag_poses = {}

    search_queue = [('camera', 0)]

    while len(search_queue) > 0:
        search_type, id = search_queue[0]
        search_queue.pop(0)

        if search_type == 'camera':
            for tag_id, pose in [(tag_id, pose) for tag_id, camera_id, pose in correspondences if camera_id == id]:
                if tag_id not in tag_poses:
                    tag_poses[tag_id] = camera_extrinsics[id] @ pose
                    search_queue.append(('tag', tag_id))
        elif search_type == 'tag':
            for camera_id, pose in [(camera_id, pose) for tag_id, camera_id, pose in correspondences if tag_id == id]:
                if camera_id not in camera_extrinsics:
                    camera_extrinsics[camera_id] = tag_poses[id] @ np.linalg.inv(pose)
                    search_queue.append(('camera', camera_id))

    if save_path is not None:
        save_dict = {}
        for camera_id, e in camera_extrinsics.items():
            save_dict[camera_names[camera_id]] = e.tolist()
        with open(save_path, 'w') as file:
            yaml.dump(save_dict, file)

    return_dict = {}
    for camera_id, e in camera_extrinsics.items():
        return_dict[camera_names[camera_id]] = e
        overlays[camera_id] = add_border(overlays[camera_id], width=3)

    preview_img = collage_imgs(overlays)

    return return_dict, preview_img


if __name__ == '__main__':
    scene_name = 'scene_2210212236'
    master_image_id = 0

    camera_names = get_camera_names(f'{dataset_path}/{scene_name}')
    camera_names_image_params = {}
    for i, camera_name in enumerate(camera_names):
        camera_path = f'{dataset_path}/{scene_name}/{camera_name}'
        image_path = f'{camera_path}/rgb/{master_image_id:06}.png'
        im = cv2.imread(image_path)
        intrinsics = load_intrinsics(f'{camera_path}/camera_meta.yml')
        camera_params = (intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        camera_names_image_params[camera_name] = (im, camera_params)

    ext, preview_img = process_extrinsics(camera_names_image_params, tag_size=0.08,
                                             save_path=f'{dataset_path}/{scene_name}/extrinsics.yml')
    print(ext)
    cv2.imshow('ext', preview_img)
    cv2.waitKey(0)
