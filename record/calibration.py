import os
import shutil
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

from dataset_tools.config import dataset_path, resolution_width, resolution_height
from dataset_tools.process.calculate_extrinsics import process_extrinsics
from dataset_tools.process.helpers import collage_imgs

# settings
frame_rate = 6  # fps
scene_name = f'scene_{datetime.now().strftime("%y%m%d%H%M")}_ext'
num_frame_for_each_angle = 50


if __name__ == '__main__':
    class Device:
        def __init__(self, camera_path, pipeline, pipeline_profile):
            self.camera_path = camera_path
            self.pipeline = pipeline
            self.pipeline_profile = pipeline_profile

    # Create scene and camera folders
    scene_path = f'{dataset_path}/{scene_name}'
    if os.path.exists(scene_path):
        shutil.rmtree(scene_path)
    os.mkdir(scene_path)
    print('Saving data to', scene_path)

    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

    # Enable all devices
    devices = {}
    context = rs.context()

    print(len(context.devices), 'have been found')
    assert len(context.devices) > 0
    for i, device in enumerate(context.devices):
        if device.get_info(rs.camera_info.name).lower() != 'platform camera':
            print(device)
            serial = device.get_info(rs.camera_info.serial_number)
            device_name = f'camera_{i+1:02d}_{serial}'
            camera_path = f'{scene_path}/{device_name}'
            os.mkdir(camera_path)

            # Enable D400 device
            pipeline = rs.pipeline()
            rs_config.enable_device(serial)
            pipeline_profile = pipeline.start(rs_config)

            devices[device_name] = Device(camera_path, pipeline, pipeline_profile)

    print(f'{len(devices)} devices are enabled')

    # preview or record images for calibration
    calibrate = False
    num_frame = 0
    calibrate_frame = 0
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

        if calibrate:
            num_frame += 1
            calibrate_frame += 1
            if calibrate_frame == num_frame_for_each_angle:
                print(f'Saved {num_frame_for_each_angle} images')
                print('press E to record from another angle')
                print('press Q to quit recording and start calibration')
                calibrate = False
                calibrate_frame = 0

        preview_img = collage_imgs(color_images)
        cv2.namedWindow('Preview', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Preview', preview_img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if key & 0xFF == ord('e'):
            print(f'Start recording {num_frame_for_each_angle} images for calibration...')
            calibrate = True

    # Calibration
