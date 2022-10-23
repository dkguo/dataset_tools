import os
import shutil
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

from config import dataset_path, resolution_width, resolution_height
from process.calculate_extrinsics import process_extrinsics
from process.helpers import collage_imgs
from process.unzip_bags import unzip_bags

# settings
frame_rate = 30  # fps
april_tag_size = 0.08
scene_name = 'scene_' + datetime.now().strftime("%y%m%d%H%M")


if __name__ == '__main__':
    class Device:
        def __init__(self, pipeline, pipeline_profile, recorder):
            self.pipeline = pipeline
            self.pipeline_profile = pipeline_profile
            self.recorder = recorder


    # Create scene and camera folders
    scene_path = f'{dataset_path}/{scene_name}'
    if os.path.exists(scene_path):
        shutil.rmtree(scene_path)
    os.mkdir(scene_path)
    print('Saving data to', scene_path)

    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
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
            camera_path = f'{scene_path}/{device_name}.bag'

            # Enable D400 device
            pipeline = rs.pipeline()
            rs_config.enable_record_to_file(camera_path)
            rs_config.enable_device(serial)
            pipeline_profile = pipeline.start(rs_config)
            recorder = pipeline_profile.get_device().as_recorder()  # Obtain the streaming device and cast it to recorder type
            recorder.pause()  # Pause recording while continuing to stream

            devices[device_name] = Device(pipeline, pipeline_profile, recorder)

    print(f'{len(devices)} devices are enabled')

    # preview
    save_extrinsics = False
    while True:
        camera_names_image_params = {}
        color_images = []
        depth_images = []
        for device_name, device in devices.items():
            streams = device.pipeline_profile.get_streams()
            frameset = device.pipeline.wait_for_frames()  # frameset will be a pyrealsense2.composite_frame object

            if frameset.size() == len(streams):
                for stream in streams:
                    if stream.stream_type() == rs.stream.color:
                        color_frame = frameset.first_or_default(stream.stream_type())
                        color_image = np.asanyarray(color_frame.get_data())
                        color_images.append(color_image)
                    elif stream.stream_type() == rs.stream.depth:
                        depth_frame = frameset.first_or_default(stream.stream_type())
                        depth_image = np.asanyarray(depth_frame.get_data())
                        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3),
                                                        cv2.COLORMAP_JET)
                        depth_images.append(depth_image)
            else:
                continue

            if save_extrinsics:
                intrinsics = frameset.get_profile().as_video_stream_profile().get_intrinsics()
                fx, fy, px, py = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
                camera_names_image_params[device_name] = (color_image, (fx, fy, px, py))

        if save_extrinsics:
            ext, preview_img = process_extrinsics(camera_names_image_params, april_tag_size,
                                         save_path=f'{dataset_path}/{scene_name}/extrinsics.yml')
        else:
            preview_img = collage_imgs(color_images + depth_images)

        cv2.namedWindow('Preview', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Preview', preview_img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if key & 0xFF == ord('e'):
            save_extrinsics = not save_extrinsics

    # Start recording
    for device_name, device in devices.items():
        device.recorder.resume()
        print(device_name, 'starts recording...')
    i = 0
    prev_time = time.time()
    while True:
        cur_time = time.time()
        if i % 30 == 29:
            print(f'frame rate: {30 / (cur_time - prev_time)} fps')
            prev_time = cur_time

        for device in devices.values():
            device.pipeline.wait_for_frames()

        i += 1
