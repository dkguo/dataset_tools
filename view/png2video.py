import glob

import cv2

from config import dataset_path
from view import scene_name

if __name__ == '__main__':


    camera_paths = glob.glob(f'{dataset_path}/{scene_name}/camera_*')
    print(len(camera_paths), 'cameras have been found:')
    print(camera_paths)

    for camera_path in camera_paths:
        print(camera_path)
        color_png_paths = sorted(glob.glob(f'{camera_path}/rgb/*.png'))

        out_video = cv2.VideoWriter(f'{camera_path}/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))

        for color_png_path in color_png_paths:
            out_im = cv2.imread(color_png_path)
            out_video.write(out_im)

        out_video.release()