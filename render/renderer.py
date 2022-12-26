import time

import cv2
import numpy as np
import torch
from transforms3d.quaternions import mat2quat

from dataset_tools.config import dataset_path, resolution_width, resolution_height
from dataset_tools.loaders import load_intrinsics
from dataset_tools.renderer import overlay_imgs
from ycb_renderer import YCBRenderer

models_path = f'{dataset_path}/models'
ycb_model_names = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
    "026_sponge"
]

obj_model_paths = {}
obj_texture_paths = {}
for model_name in ycb_model_names:
    obj_id = int(model_name[:3])
    obj_model_paths[obj_id] = f'{models_path}/{model_name}/textured_simple.obj'
    obj_texture_paths[obj_id] = f'{models_path}/{model_name}/texture_map.png'


def load_objs(renderer, obj_ids=None):
    for obj_id in obj_ids:
        renderer.obj_ids.append(obj_id)
        renderer.load_object(obj_model_paths[obj_id], obj_texture_paths[obj_id])


def create_scene(intrinsics, obj_ids=None, width=resolution_width, height=resolution_height):
    renderer = YCBRenderer(width, height, render_marker=False)
    renderer.set_camera_default()
    renderer.set_projection_matrix(resolution_width, resolution_height,
                                   intrinsics[0, 0], intrinsics[1, 1],
                                   intrinsics[0, 2], intrinsics[1, 2], 0.01, 10)
    load_objs(renderer, obj_ids)
    return renderer


def render_obj_pose(renderer, list_id_pose, width=resolution_width, height=resolution_height, unit='mm'):
    poses = []
    obj_idxes = []  # different from obj_ids; is the load squence in renderer
    pose_v = np.zeros((7,))
    for obj_id, pose in list_id_pose:
        t_v = pose[:3, 3]
        q_v = mat2quat(pose[:3, :3])
        pose_v[:3] = t_v
        pose_v[3:] = q_v
        poses.append(pose_v.copy())

        if obj_id not in renderer.obj_ids:
            load_objs(renderer, [obj_id])

        obj_i = renderer.obj_ids.index(obj_id)
        obj_idxes.append(obj_i)

    # render
    frame_cuda = torch.cuda.FloatTensor(int(height), int(width), 4)
    seg_cuda = torch.cuda.FloatTensor(int(height), int(width), 4)
    pc_cuda = torch.cuda.FloatTensor(int(height), int(width), 4)
    renderer.set_poses(poses)
    renderer.render(obj_idxes, frame_cuda, seg_cuda, pc2_tensor=pc_cuda)

    frame_cuda = frame_cuda.flip(0)
    frame_cuda = frame_cuda[:, :, :3].float()
    frame_cuda = frame_cuda.permute(2, 0, 1).unsqueeze(0)

    return frame_cuda


if __name__ == '__main__':
    # test_path = '/media/gdk/Hard_Disk/Datasets/old_kitchen_data/multiple_depth_cameras/1652915948/752112070756'
    # test_img_name = '000036_color.png'
    # list_id_poses = [(3, np.array([[0.50324144, 0.86231973, -0.05614924, -0.00303771],
    #                                [0.78285323, -0.48244995, -0.39291584, -0.0212927],
    #                                [-0.36590828, 0.15377491, -0.9178586, 1.0443474],
    #                                [0, 0, 0, 1]]))]
    #
    # intrinsics = load_intrinsics(f'{test_path}/meta.yml')
    # test_im = cv2.imread(f'{test_path}/{test_img_name}')
    #
    # # render objs with pyrender
    # renderer = create_scene(intrinsics)
    # t = time.time()
    # rendered_im = render_obj_pose(renderer, list_id_poses)
    # print(time.time() - t)
    # rendered_im = overlay_imgs(test_im, rendered_im)
    # cv2.imshow('original', test_im)
    # cv2.imshow('render', rendered_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    renderer = YCBRenderer(resolution_width, resolution_height, render_marker=False)
    renderer.load_objects([obj_model_paths[24]], [obj_texture_paths[24]])
    pose = np.array([0, -0.1, 0.05, 0.6582951, 0.03479896, -0.036391996, -0.75107396])
    theta = 0
    phi = 0
    psi = 0
    r = 1
    cam_pos = [np.sin(theta) * np.cos(phi) * r, np.sin(phi) * r, np.cos(theta) * np.cos(phi) * r]
    # renderer.set_camera(cam_pos, [0, 0, 0], [0, 1, 0])
    # renderer.set_fov(40)
    renderer.set_poses([pose])
    renderer.set_light_pos(cam_pos)
    print(cam_pos)
    renderer.set_light_color([1.5, 1.5, 1.5])

    tensor = torch.cuda.FloatTensor(resolution_height, resolution_width, 4)
    tensor2 = torch.cuda.FloatTensor(resolution_height, resolution_width, 4)
    pc_tensor = torch.cuda.FloatTensor(resolution_height, resolution_width, 4)

    t = time.time()
    renderer.render([0], tensor, seg_tensor=tensor2, pc2_tensor=pc_tensor)

    img_np = tensor.flip(0).data.cpu().numpy().reshape(resolution_height, resolution_width, 4)
    print(time.time() - t)
    cv2.imshow('test', cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)