import glob
import multiprocessing
import os
from datetime import datetime
from multiprocessing import Pool

import pyrealsense2 as rs
import numpy as np
import cv2
import yaml

from bop_toolkit.bop_toolkit_lib.inout import save_im, save_depth
from config import dataset_path
from process import scene_name
from record.record_multiple_d435 import resolution_width, resolution_height


def unzip_bag(camera_num, bag_path):
    camera_serial = os.path.basename(bag_path)[:-4]
    print('processing', bag_path)
    camera_path = f'{dataset_path}/{scene_name}/camera_{camera_num:02d}_{camera_serial}'

    # create folders
    if not os.path.exists(camera_path):
        os.mkdir(camera_path)
        os.mkdir(f'{camera_path}/rgb')
        os.mkdir(f'{camera_path}/depth')

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_path)

    config.enable_stream(rs.stream.depth, rs.format.z16)
    config.enable_stream(rs.stream.color, rs.format.bgr8)

    profile = pipeline.start(config)

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)

    # save intrinsics
    frameset = pipeline.wait_for_frames()
    frameset = align.process(frameset)
    prev_time = frameset.get_timestamp() / 1000

    for stream, frame in frameset.items():
        if stream == rs.stream.color:
            intrinsics = frame.get_profile().as_video_stream_profile().get_intrinsics()

    fx, fy, px, py = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

    with open(f'{camera_path}/camera_meta.yml', 'w') as file:
        assert True
        save_str = {'INTRINSICS': [fx, 0.0, px, 0.0, fy, py, 0.0, 0.0, 1.0],
                    'DEPTH_UNIT': 'mm',
                    'FRAME_WIDTH': resolution_width,
                    'FRAME_HEIGHT': resolution_height}
        yaml.dump(save_str, file)

    return


    frame_num = 0
    while True:
        frameset = pipeline.wait_for_frames()
        frameset = align.process(frameset)
        timestamp = frameset.get_timestamp() / 1000
        if timestamp < prev_time:
            break
        prev_time = timestamp

        color_frame = frameset.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = frameset.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())

        # save imgs
        color_image_path = f'{camera_path}/rgb/{frame_num:06d}.png'
        save_im(color_image_path, color_image[:, :, ::-1])
        depth_image_path = f'{camera_path}/depth/{frame_num:06d}.png'
        save_depth(depth_image_path, depth_image)

        # preview
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)
        image = np.hstack((color_image, depth_colormap))
        text = f'Frame: {frame_num}, Time: {datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")}'
        image = cv2.putText(image, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        cv2.imshow(f'camera_{camera_num:02d}_{camera_serial}', image)
        cv2.waitKey(1)

        frame_num += 1


if __name__ == '__main__':
    bag_paths = glob.glob(f'{dataset_path}/{scene_name}/*.bag')
    print(len(bag_paths), 'have been found:')
    print(bag_paths)

    # args = []
    # for i, bag_path in enumerate(bag_paths):
    #     args.append((i + 1, bag_path))
    #
    # multiprocessing.set_start_method('spawn')
    # with Pool() as pool:
    #     pool.starmap(unzip_bag, args)
