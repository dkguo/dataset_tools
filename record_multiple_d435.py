import os
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

from config import dataset_path


# settings
from process.apriltag_detection import detect_april_tag

resolution_width = 640  # pixels
resolution_height = 480  # pixels
frame_rate = 15  # fps
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
    os.mkdir(scene_path)
    uid = os.environ.get("SUDO_UID")
    gid = os.environ.get("SUDO_GID")
    if uid:
        os.chown(scene_path, int(uid), int(gid))
    print('Saving data to', scene_path)

    try:
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

        # Enable all devices
        devices = {}
        context = rs.context()

        print(len(context.devices), 'have been found')
        assert len(context.devices) > 0
        for device in context.devices:
            print(device)
            if device.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = device.get_info(rs.camera_info.serial_number)

                camera_path = f'{scene_path}/{serial}.bag'

                # Enable D400 device
                pipeline = rs.pipeline()
                rs_config.enable_record_to_file(camera_path)
                rs_config.enable_device(serial)
                pipeline_profile = pipeline.start(rs_config)
                recorder = pipeline_profile.get_device().as_recorder()  # Obtain the streaming device and cast it to recorder type
                recorder.pause()  # Pause recording while continuing to stream

                devices[serial] = Device(pipeline, pipeline_profile, recorder)

        print(f'{len(devices)} devices are enabled')

        # preview
        for device in devices.values():
            while True:
                streams = device.pipeline_profile.get_streams()
                frameset = device.pipeline.wait_for_frames()  # frameset will be a pyrealsense2.composite_frame object

                if frameset.size() == len(streams):
                    for stream in streams:
                        if stream.stream_type() == rs.stream.color:
                            color_frame = frameset.first_or_default(stream.stream_type())
                            color_image = np.asanyarray(color_frame.get_data())
                        elif stream.stream_type() == rs.stream.depth:
                            depth_frame = frameset.first_or_default(stream.stream_type())
                            depth_image = np.asanyarray(depth_frame.get_data())
                else:
                    continue

                prev_time = frameset.get_timestamp() / 1000
                intrinsics = frameset.get_profile().as_video_stream_profile().get_intrinsics()
                fx, fy, px, py = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

                pose, color_im_w_tag = detect_april_tag(color_image, (fx, fy, px, py), april_tag_size)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)
                images = np.hstack((color_im_w_tag, depth_colormap))

                cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
                cv2.imshow('Camera', images)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

        # Start recording
        for serial, device in devices.items():
            device.recorder.resume()
            print(serial, 'starts recording...')
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

    except KeyboardInterrupt:
        print("The program was interupted by the user. Closing the program...")

        serials = sorted(devices.keys())
        # for serial, device in devices.items():
        for serial in serials:
            devices[serial].pipeline.stop()
            # device.pipeline.stop()
            print(serial, 'stopped')

        print("All devices are stopped")
        print('Data is saved to', scene_path)
