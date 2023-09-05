import multiprocessing
import os
import shutil
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

from dataset_tools.bop_toolkit.bop_toolkit_lib.inout import save_im, save_depth
from dataset_tools.config import dataset_path, resolution_width, resolution_height
from dataset_tools.record.step_2_unzip_bags import save_intrinsics
from dataset_tools.view.helpers import collage_imgs


class Device:
    def __init__(self, pipeline, pipeline_profile, recorder):
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile
        self.recorder = recorder


class D435s:
    def __init__(self, scene_name=None, frame_rate=30, init_recorder=True):
        # Create scene and camera folders
        self.scene_name = 'scene_' + datetime.now().strftime("%y%m%d%H%M%S") if scene_name is None else scene_name
        self.scene_path = f'{dataset_path}/{self.scene_name}'
        os.makedirs(self.scene_path, exist_ok=True)
        print('Saving data to', self.scene_path)

        # Enable all devices
        self.devices = {}
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
        context = rs.context()
        print(len(context.devices), 'have been found')
        assert len(context.devices) > 0
        for i, device in enumerate(context.devices):
            if device.get_info(rs.camera_info.name).lower() != 'platform camera':
                print(device)
                serial = device.get_info(rs.camera_info.serial_number)
                device_name = f'camera_{i + 1:02d}_{serial}'
                camera_path = f'{self.scene_path}/{device_name}.bag'

                # Enable D400 device
                pipeline = rs.pipeline()
                if init_recorder:
                    rs_config.enable_record_to_file(camera_path)
                rs_config.enable_device(serial)
                pipeline_profile = pipeline.start(rs_config)
                if init_recorder:
                    # Obtain the streaming device and cast it to recorder type
                    recorder = pipeline_profile.get_device().as_recorder()
                    recorder.pause()  # Pause recording while continuing to stream
                else:
                    recorder = None

                self.devices[device_name] = Device(pipeline, pipeline_profile, recorder)

        print(f'{len(self.devices)} devices are enabled')

    def _get_images(self, align_depth_to_color=False):
        color_images = []
        depth_images = []
        colorized_depth_images = []
        framesets = []
        align = rs.align(rs.stream.color) if align_depth_to_color else None
        for device_name, device in self.devices.items():
            streams = device.pipeline_profile.get_streams()
            frameset = device.pipeline.wait_for_frames()  # frameset will be a pyrealsense2.composite_frame object
            if align_depth_to_color:
                frameset = align.process(frameset)
            framesets.append(frameset)
            if frameset.size() == len(streams):
                for stream in streams:
                    if stream.stream_type() == rs.stream.color:
                        color_frame = frameset.first_or_default(stream.stream_type())
                        color_image = np.asanyarray(color_frame.get_data())
                        color_images.append(color_image)
                    elif stream.stream_type() == rs.stream.depth:
                        depth_frame = frameset.first_or_default(stream.stream_type())
                        depth_image = np.asanyarray(depth_frame.get_data())
                        depth_images.append(depth_image)
                        colorized_depth_images.append(cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3),
                                                                        cv2.COLORMAP_JET))
        return color_images, depth_images, colorized_depth_images, framesets

    def preview(self):
        while True:
            color_images, _, depth_images, _ = self._get_images()
            preview_img = collage_imgs(color_images + depth_images)
            cv2.namedWindow('Preview', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Preview', preview_img)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            elif key & 0xFF == ord('c'):
                self.capture()

    def capture(self, view=True):
        frame_num = datetime.now().strftime("%H%M%S")
        color_images, depth_images, colorized_depth_images, framesets = self._get_images(align_depth_to_color=True)

        for device_name, frameset in zip(self.devices.keys(), framesets):
            camera_path = f'{self.scene_path}/{device_name}'
            if not os.path.exists(camera_path):
                os.mkdir(camera_path)
                os.mkdir(f'{camera_path}/rgb')
                os.mkdir(f'{camera_path}/depth')
                save_intrinsics(camera_path, frameset)

        if view:
            preview_img = collage_imgs(color_images + colorized_depth_images)
            cv2.namedWindow('Capture', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Capture', preview_img)

            key = cv2.waitKey()
            cv2.destroyAllWindows()
            if key & 0xFF == ord('q'):
                return
            elif key & 0xFF == ord('r'):
                return self.capture()

        for device_name, color_image, depth_image in zip(self.devices.keys(), color_images, depth_images):
            camera_path = f'{self.scene_path}/{device_name}'
            save_im(f'{camera_path}/rgb/{frame_num}.png', color_image[:, :, ::-1])
            save_depth(f'{camera_path}/depth/{frame_num}.png', depth_image)
        print(f'Frame {frame_num} is saved')
        return int(frame_num)

    def record(self):
        for device_name, device in self.devices.items():
            device.recorder.resume()
            print(device_name, 'starts recording...')
        i = 0
        prev_time = time.time()
        while True:
            cur_time = time.time()
            if i % 30 == 29:
                print(f'frame rate: {30 / (cur_time - prev_time)} fps')
                prev_time = cur_time
            for device in self.devices.values():
                device.pipeline.wait_for_frames()
            i += 1


if __name__ == '__main__':
    cameras = D435s()
    cameras.preview()
    cameras.record()
