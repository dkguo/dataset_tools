import csv

import cv2

from config import dataset_path
from process.helpers import add_texts, add_green_texts


if __name__ == '__main__':
    scene_name = 'scene_2210232307_01'
    # load primitive csv
    with open(f'{dataset_path}/{scene_name}/primitives.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        header = next(spamreader)
        frames_primitives = []
        for row in spamreader:
            frames_primitives.append(row)

    print(frames_primitives)

    # process text to add on each frame
    frame_text = {}
    for start_frame, end_frame, primitive in frames_primitives:
        for frame in range(int(start_frame), int(end_frame) + 1):
            if frame in frame_text:
                frame_text[frame].append(primitive)
            else:
                frame_text[frame] = [primitive]

    # load video, add text, and make new video
    cap = cv2.VideoCapture(f'{dataset_path}/{scene_name}/video.mp4')
    assert cap.isOpened()

    frame_rate = 10
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frame_rate, width, height)
    out = cv2.VideoWriter(f'{dataset_path}/{scene_name}/video_primitives.mp4',  cv2.VideoWriter_fourcc(*'mp4v'),
                          frame_rate, (width, height))

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if i in frame_text:
            print(frame_text[i])
            frame = add_green_texts(frame, frame_text[i], start_xy=[50, 100], thickness=10, font_scale=2.5)

        out.write(frame)
        i += 1

    out.release()

