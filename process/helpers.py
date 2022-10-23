import cv2
import numpy as np


def collage_imgs(ims, num_rows=2):
    """
    num_rows: only works when len(imgs) >=7
    """
    imgs = ims.copy()
    if 7 <= len(imgs) <= 9:
        if num_rows == 2:
            for i in range(len(imgs), 8):
                imgs.append(np.zeros_like(imgs[0]))
        elif num_rows == 3:
            for i in range(len(imgs), 9):
                imgs.append(np.zeros_like(imgs[0]))

    if len(imgs) == 2:
        img = np.concatenate(imgs, axis=1)
    elif len(imgs) == 3:
        img = np.concatenate(imgs, axis=1)
    elif len(imgs) == 4:
        h_im_1 = np.concatenate([imgs[0], imgs[1]], axis=1)
        h_im_2 = np.concatenate([imgs[2], imgs[3]], axis=1)
        img = np.concatenate([h_im_1, h_im_2], axis=0)
    elif len(imgs) == 6:
        h_ims = []
        for i in range(2):
            h_ims.append(np.concatenate([imgs[3 * i], imgs[3 * i + 1], imgs[3 * i + 2]], axis=1))
        img = np.concatenate(h_ims, axis=0)
    elif len(imgs) == 8:
        h_im_1 = np.concatenate([imgs[0], imgs[1], imgs[2], imgs[3]], axis=1)
        h_im_2 = np.concatenate([imgs[4], imgs[5], imgs[6], imgs[7]], axis=1)
        img = np.concatenate([h_im_1, h_im_2], axis=0)
    elif len(imgs) == 9:
        h_ims = []
        for i in range(3):
            h_ims.append(np.concatenate([imgs[3*i], imgs[3*i+1], imgs[3*i+2]], axis=1))
        img = np.concatenate(h_ims, axis=0)
    elif len(imgs) == 16:
        h_im_1 = collage_imgs(imgs[:8])
        h_im_2 = collage_imgs(imgs[8:])
        img = np.concatenate([h_im_1, h_im_2], axis=0)
    else:
        assert True, f'{len(imgs)} images is not supported'
    return img


def add_border(img, color=1, width=10):
    """
    color: 0, 1, 2 -> r, g, b
    """
    border = np.zeros_like(img)
    border[:, :, color] = 255
    border[width:-width, width:-width, :] = img[width:-width, width:-width, :]
    return border


def add_texts(img, text_list, start_xy=(30, 30), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    text_list: [(text, color)], color=(0, 255, 0)
    """
    x, y = start_xy
    for i, (text, color) in enumerate(text_list):
        img = cv2.putText(img, text, (x, y + i * 30), font, 1, color, thickness=thickness)
    return img
