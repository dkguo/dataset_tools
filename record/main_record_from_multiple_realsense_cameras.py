import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from record.realsense_device_manager import DeviceManager


# settings
resolution_width = 640  # pixels
resolution_height = 480  # pixels
frame_rate = 15  # fps
dataset_path = '/Downloads/data'
scene_name = int(time.time())


try:
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

    device_manager = DeviceManager(rs.context(), rs_config)
    device_manager.enable_all_devices()

    assert (len(device_manager._available_devices) > 0)

    frames = device_manager.poll_frames()

    # preview
    for k in frames.keys():
        while True:
            frames = device_manager.poll_frames()
            depth_image = np.asanyarray(frames[k][rs.stream.depth].get_data())
            color_image = np.asanyarray(frames[k][rs.stream.color].get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
            cv2.imshow('Camera', images)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    # create scene folder
    scene_path = f'{dataset_path}/{scene_name}'
    print(scene_path)
    os.mkdir(scene_path)

    # create camera folders
    dict_cameras_k_path = {}
    for i, k in enumerate(frames.keys()):
        camera_name = f'camera_{i+1:02d}'
        camera_path = f'{scene_path}/{camera_name}'
        os.mkdir(camera_path)
        os.mkdir(f'{camera_path}/rgb')
        os.mkdir(f'{camera_path}/depth')
        dict_cameras_k_path[k] = camera_path

    # save intrinsics
    intrinsics_devices = device_manager.get_device_intrinsics(frames)
    for k in intrinsics_devices.keys():
        intrinsics = intrinsics_devices[k][rs.stream.color]
        fx, fy, px, py = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        print(fx, fy, px, py)

        camera_path = dict_cameras_k_path[k]
        with open(f'{camera_path}/intrinsics.json', 'w') as file:
            intrinsics = [fx, 0.0, px, 0.0, fy, py, 0.0, 0.0, 1.0]
            json.dump(intrinsics, file)

    # save frames
    i = 0
    print("recording...")
    time_list = []
    prev_time = time.time()
    while True:
        cur_time = time.time()
        time_list.append(cur_time)
        if i % 30 == 29:
            print(f'frame rate: {30 / (cur_time - prev_time)} fps')
            prev_time = cur_time
        frames = device_manager.poll_frames()
        for k in frames.keys():
            color_image = np.asanyarray(frames[k][rs.stream.color].get_data())
            depth_image = np.asanyarray(frames[k][rs.stream.depth].get_data())

            camera_path = dict_cameras_k_path[k]
            np.save(f'{camera_path}/rgb/{i:06d}.npy', color_image)
            np.save(f'{camera_path}/depth/{i:06d}.npy', depth_image)

        i += 1

except KeyboardInterrupt:
    with open(f'{scene_path}/time.json', 'w') as file:
        json.dump(time_list, file)

    print("The program was interupted by the user. Closing the program...")

finally:
    device_manager.disable_streams()
    cv2.destroyAllWindows()
