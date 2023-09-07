import cv2
import numpy as np
import apriltag
import collections

from tqdm import tqdm

from dataset_tools.config import dataset_path
from dataset_tools.utils.camera_parameter import intr2param, load_cameras_intrisics, load_cameras_extrinsics
from dataset_tools.utils.image import load_imgs_across_cameras
from dataset_tools.utils.name import get_camera_names
from dataset_tools.utils.image import collage_imgs
from modules.object_pose_detection.multiview_voting import combine_poses

apriltag_detect_error_thres = 0.07


def draw_pose(overlay, camera_params, tag_size, pose, z_sign=1, color=(0, 255, 0)):
    opoints = np.array([
        -1, -1, 0,
        1, -1, 0,
        1, 1, 0,
        -1, 1, 0,
        -1, -1, -2 * z_sign,
        1, -1, -2 * z_sign,
        1, 1, -2 * z_sign,
        -1, 1, -2 * z_sign,
    ]).reshape(-1, 1, 3) * 0.5 * tag_size

    edges = np.array([
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        0, 4,
        1, 5,
        2, 6,
        3, 7,
        4, 5,
        5, 6,
        6, 7,
        7, 4
    ]).reshape(-1, 2)

    fx, fy, cx, cy = camera_params

    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)

    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]

    dcoeffs = np.zeros(5)

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)

    ipoints = np.round(ipoints).astype(int)

    ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

    for i, j in edges:
        cv2.line(overlay, ipoints[i], ipoints[j], color, 1, 16)


def draw_pose_axes(overlay, camera_params, tag_size, pose, center=None):
    fx, fy, cx, cy = camera_params
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)

    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]

    dcoeffs = np.zeros(5)

    opoints = np.float32([[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]]).reshape(-1, 3) * tag_size

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)
    ipoints = np.round(ipoints).astype(int)

    cpoints = np.float32([[0, 0, 0]]).reshape(-1, 3) * tag_size
    cpoints, _ = cv2.projectPoints(cpoints, rvec, tvec, K, dcoeffs)
    cpoints = np.round(cpoints).astype(int)

    center = tuple(cpoints[0].ravel())

    cv2.line(overlay, center, tuple(ipoints[0].ravel()), (0, 0, 255), 2)
    cv2.line(overlay, center, tuple(ipoints[1].ravel()), (0, 255, 0), 2)
    cv2.line(overlay, center, tuple(ipoints[2].ravel()), (255, 0, 0), 2)


def annotate_detection(overlay, detection, center):
    text = str(detection.tag_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    tag_size_px = np.sqrt((detection.corners[1][0] - detection.corners[0][0]) ** 2 + \
                          (detection.corners[1][1] - detection.corners[0][1]) ** 2)
    font_size = tag_size_px / 22
    text_size = cv2.getTextSize(text, font, font_size, 2)[0]
    tag_center = [detection.center[0], detection.center[1]]
    text_x = int(tag_center[0] - text_size[0] / 2)
    text_y = int(tag_center[1] + text_size[1] / 2)
    cv2.putText(overlay, text, (text_x, text_y), font, font_size, (0, 255, 255), 2)


def detect_april_tag(orig, camera_params, tag_size, visualize=False, save_path=None, verbose=False):
    if len(orig.shape) == 3:
        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)

    detector = apriltag.Detector()
    detections, dimg = detector.detect(gray, return_image=True)

    num_detections = len(detections)
    if verbose:
        print(f'Detected {num_detections} tags')

    if num_detections == 0:
        overlay = orig
    elif len(orig.shape) == 3:
        overlay = orig // 2 + dimg[:, :, None] // 2
    else:
        overlay = orig // 2 + dimg // 2

    poses = []
    for i, detection in enumerate(detections):
        if verbose:
            print()
            print('Detection {} of {}:'.format(i + 1, num_detections))
            print(detection.tostring(indent=2))

        if camera_params is not None:
            pose, e0, ef = detector.detection_pose(detection, camera_params, tag_size)
            poses.append((detection.tag_id, pose, ef))
            draw_pose(overlay, camera_params, tag_size, pose)
            draw_pose_axes(overlay, camera_params, tag_size, pose, detection.center)
            annotate_detection(overlay, detection, tag_size)

            if verbose:
                print(detection.tostring(collections.OrderedDict([('Pose', pose),
                                                                  ('InitError', e0), ('FinalError', ef)]), indent=2))

    if visualize:
        cv2.imshow('apriltag', overlay)
        while cv2.waitKey(5) < 0:  # Press any key to load subsequent image
            continue
        cv2.destroyAllWindows()

    if save_path is not None:
        cv2.imwrite(save_path, overlay)

    return poses, overlay


def verify_calibration(cameras_img, cameras_intr, cameras_ext, tag_size=0.08):
    # register all tags
    tags = {}
    for camera_name, image in cameras_img.items():
        camera_params = intr2param(cameras_intr[camera_name])
        poses_errs, overlay = detect_april_tag(image, camera_params, tag_size)

        for tag_id, pose, error in poses_errs:
            if error < apriltag_detect_error_thres and tag_id not in tags:
                tags[tag_id] = (camera_name, pose)

    # plot all tags
    for camera_name, image in cameras_img.items():
        camera_params = intr2param(cameras_intr[camera_name])
        for tag_camera_name, pose in tags.values():
            if camera_name == tag_camera_name:
                draw_pose(image, camera_params, tag_size, pose, color=(0, 0, 255))
            else:
                tag_pose = np.linalg.inv(cameras_ext[camera_name]) @ cameras_ext[tag_camera_name] @ pose
                draw_pose(image, camera_params, tag_size, tag_pose)

    return cameras_img


def get_tag_poses(scene_name, frames, target_tag_id, tag_size, mode='best'):
    print('Extracting tag pose...')

    scene_path = f'{dataset_path}/{scene_name}'

    cameras_intr = load_cameras_intrisics(scene_name)
    cameras_ext = load_cameras_extrinsics(scene_name)

    if mode == 'best':
        min_ef = np.Inf
        best_pose = None
        for frame in tqdm(frames, disable=True if len(frames) < 10 else False):
            imgs = load_imgs_across_cameras(scene_path, get_camera_names(scene_path), f'{frame:06d}.png')
            for camera, img in zip(get_camera_names(scene_path), imgs):
                intr = cameras_intr[camera]
                param = intr2param(intr)
                poses, overlay = detect_april_tag(img, param, tag_size)
                for tag_id, pose, ef in poses:
                    if tag_id == target_tag_id and ef < min_ef:
                        min_ef = ef
                        best_pose = (camera, pose)
        camera, pose = best_pose
    elif mode == 'combine':
        poses = []
        c0 = get_camera_names(scene_path)[0]
        for frame in tqdm(frames, disable=True if len(frames) < 10 else False):
            imgs = load_imgs_across_cameras(scene_path, get_camera_names(scene_path), f'{frame:06d}.png')
            for camera, img in zip(get_camera_names(scene_path), imgs):
                intr = cameras_intr[camera]
                param = intr2param(intr)
                _poses, overlay = detect_april_tag(img, param, tag_size)
                _poses = [pose for tag_id, pose, ef in _poses if tag_id == target_tag_id]
                _poses = [np.linalg.inv(cameras_ext[c0]) @ cameras_ext[camera] @ pose for pose in _poses]
                poses.extend(_poses)
        pose = combine_poses(poses, thres_dist=0.01, thres_q_sim=0.99, verbose=True)
        camera = c0
    else:
        raise ValueError(f'Unknown mode: {mode}')

    # change to right pose
    rm = np.zeros((4, 4))
    rm[0, 0] = 1
    rm[1, 1] = -1
    rm[2, 2] = -1
    rm[3, 3] = 1
    tag_pose = pose @ rm

    rm = np.zeros((4, 4))
    rm[0, 1] = 1
    rm[1, 0] = -1
    rm[2, 2] = 1
    rm[3, 3] = 1
    tag_pose = tag_pose @ rm

    return cameras_ext[camera] @ tag_pose


def view_april_tag_pose(tag_pose, imgs, cameras_intr, camera_ext, tag_size, render_sugar_box=False):
    # original tags
    overlays = []
    for img, intr in zip(imgs, cameras_intr.values()):
        poses, overlay = detect_april_tag(img, intr2param(intr), tag_size)
        overlays.append(overlay)
    overlay = collage_imgs(overlays)
    cv2.imshow('orignal tags', overlay)

    for img, intr, ext in zip(imgs, cameras_intr.values(), camera_ext.values()):
        pose = np.linalg.inv(ext) @ tag_pose
        camera_params = intr2param(intr)
        draw_pose(img, camera_params, tag_size, pose)
        draw_pose_axes(img, camera_params, tag_size, pose)
    preview = collage_imgs(imgs)
    cv2.imshow('verify', preview)

    key = cv2.waitKey(0)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    imagepath = '/home/gdk/Documents/data/1652826411/827312071624/000000_color.png'
    camera_params = (765.00, 764.18, 393.72, 304.66)
    tag_size = 0.06
    detect_april_tag(imagepath, camera_params, tag_size, visualize=True, save_path=None, verbose=True)
