import cv2
import numpy as np
import apriltag
import collections

from dataset_tools.loaders import intr2param

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


if __name__ == '__main__':
    imagepath = '/home/gdk/Documents/data/1652826411/827312071624/000000_color.png'
    camera_params = (765.00, 764.18, 393.72, 304.66)
    tag_size = 0.06
    detect_april_tag(imagepath, camera_params, tag_size, visualize=True, save_path=None, verbose=True)
