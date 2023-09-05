import json
import os
import shutil
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from simple_parsing import ArgumentParser

from dataset_tools.config import dataset_path, resolution_width, resolution_height
from dataset_tools.utils import load_cameras_intrisics, load_cameras_extrinsics
from dataset_tools.view.helpers import collage_imgs, add_border
from dataset_tools.record.multical.multical.app.calibrate import Calibrate
from dataset_tools.record.apriltag_detection import verify_calibration


class Device:
    def __init__(self, camera_path, pipeline, pipeline_profile):
        self.camera_path = camera_path
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile


def create_folders(scene_path):
    """Create scene and camera folders"""
    if os.path.exists(scene_path):
        shutil.rmtree(scene_path)
    os.mkdir(scene_path)
    print('Saving data to', scene_path)


def calibration(scene_path, vis_calibration=True):
    # Calibration
    parser = ArgumentParser(prog='multical')
    parser.add_arguments(Calibrate, dest="app")
    program = parser.parse_args()

    program.app.paths.boards = f'{os.path.dirname(os.path.abspath(__file__))}/multical/example_boards/boards.yaml'
    program.app.paths.image_path = scene_path
    program.app.paths.limit_images = 2000
    program.app.vis = vis_calibration

    program.app.optimizer.iter = 5
    program.app.optimizer.loss = 'huber'
    program.app.optimizer.outlier_quantile = 0.5
    program.app.optimizer.outlier_threshold = 1.0

    program.app.optimizer.fix_intrinsic = True
    program.app.camera.calibration = f'{scene_path}/intrinsics.json'

    program.app.execute()


def generate_cameras_meta(scene_path, frame_rate):
    with open(f'{scene_path}/calibration.json') as f:
        info = json.load(f)

    intrinsics = {}
    for camera_name in info["cameras"]:
        intrinsics[camera_name] = info["cameras"][camera_name]["K"]

    extrinsics = {}
    for camera_name in info["camera_poses"]:
        R = info["camera_poses"][camera_name]["R"]
        T = info["camera_poses"][camera_name]["T"]
        ext = np.column_stack((R, T))
        ext = np.r_[ext, [[0, 0, 0, 1]]]
        ext = np.linalg.inv(ext)
        extrinsics[camera_name[:22]] = ext.tolist()

    meta = {'DEPTH_UNIT': 'mm',
            'FRAME_WIDTH': resolution_width,
            'FRAME_HEIGHT': resolution_height,
            'FRAME_RATE': frame_rate,
            'INTRINSICS': intrinsics,
            'EXTRINSICS': extrinsics}

    with open(f'{scene_path}/cameras_meta.yml', 'w') as file:
        yaml.dump(meta, file)

    with open(f'{scene_path}/extrinsics.yml', 'w') as file:
        yaml.dump(extrinsics, file)

    print('saved cameras_meta.yml')


if __name__ == '__main__':
    frame_rate = 6  # fps
    num_frame_for_each_angle = 10
    vis_calibration = False
    scene_name = f'scene_{datetime.now().strftime("%y%m%d%H%M%S")}_ext'
    scene_path = f'{dataset_path}/{scene_name}'

    create_folders(scene_path)

    # Enable all devices
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
    devices = {}
    context = rs.context()
    print(len(context.devices), 'have been found')
    assert len(context.devices) > 0
    for i, device in enumerate(context.devices):
        if device.get_info(rs.camera_info.name).lower() != 'platform camera':
            print(device)
            serial = device.get_info(rs.camera_info.serial_number)
            device_name = f'camera_{i + 1:02d}_{serial}'
            camera_path = f'{scene_path}/{device_name}'
            os.makedirs(camera_path, exist_ok=True)

            # Enable D400 device
            pipeline = rs.pipeline()
            rs_config.enable_device(serial)
            pipeline_profile = pipeline.start(rs_config)

            devices[device_name] = Device(camera_path, pipeline, pipeline_profile)
    print(f'{len(devices)} devices are enabled')

    # save intrinsics
    cameras = {}
    for device_name, device in devices.items():
        frameset = device.pipeline.wait_for_frames()
        intrinsics = frameset.get_profile().as_video_stream_profile().get_intrinsics()
        fx, fy, px, py = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

        cameras[device_name] = {
            "model": "standard",
            "image_size": [resolution_width, resolution_height],
            "K": [[fx, 0.0, px], [0.0, fy, py], [0.0, 0.0, 1.0]],
            "dist": [[0.0, 0.0, 0.0, 0.0, 0.0]]
        }
    with open(f'{scene_path}/intrinsics.json', 'w') as f:
        json.dump({"cameras": cameras}, f)

    # preview or record images for calibration
    calibrate = False
    num_frame = 0
    calibrate_frame = 0
    print('press E to start recording, press Q to start calibration')
    while True:
        color_images = []
        for device_name, device in devices.items():
            streams = device.pipeline_profile.get_streams()
            frameset = device.pipeline.wait_for_frames()

            if frameset.size() == len(streams):
                for stream in streams:
                    if stream.stream_type() == rs.stream.color:
                        color_frame = frameset.first_or_default(stream.stream_type())
                        color_image = np.asanyarray(color_frame.get_data())
                        color_images.append(color_image)
                        if calibrate:
                            cv2.imwrite(f'{device.camera_path}/{num_frame:06d}.png', color_image)

        preview_img = collage_imgs(color_images)

        if calibrate:
            num_frame += 1
            calibrate_frame += 1
            preview_img = add_border(preview_img)
            if calibrate_frame == num_frame_for_each_angle:
                print(f'Saved {num_frame_for_each_angle} images')
                print('press E to record from another angle')
                print('press Q to quit recording and start calibration')
                calibrate = False
                calibrate_frame = 0

        cv2.namedWindow('Preview', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Preview', preview_img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if key & 0xFF == ord('e'):
            print(f'Start recording {num_frame_for_each_angle} images for calibration...')
            calibrate = True

    calibration(scene_path, vis_calibration)
    generate_cameras_meta(scene_path, frame_rate)

    # verify calibration using april tag cube
    cameras_intr = load_cameras_intrisics(scene_name)
    cameras_ext = load_cameras_extrinsics(scene_name)
    while True:
        cameras_img = {}
        for device_name, device in devices.items():
            streams = device.pipeline_profile.get_streams()
            frameset = device.pipeline.wait_for_frames()

            if frameset.size() == len(streams):
                for stream in streams:
                    if stream.stream_type() == rs.stream.color:
                        color_frame = frameset.first_or_default(stream.stream_type())
                        color_image = np.asanyarray(color_frame.get_data())
                        cameras_img[device_name] = color_image

        cameras_img = verify_calibration(cameras_img, cameras_intr, cameras_ext, tag_size=0.06)
        preview = collage_imgs(list(cameras_img.values()))
        cv2.imshow('verify', preview)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
