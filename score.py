import csv
import json

import cv2
import imageio
import numpy as np

from dataset_tools.bop_toolkit.bop_toolkit_lib import pose_error, misc, inout, renderer
from dataset_tools.bop_toolkit.bop_toolkit_lib.inout import load_depth
from config import ply_model_paths, models_info_path, obj_model_paths, dataset_path, resolution_width, resolution_height
from loaders import load_ground_truth, load_bop_est_pose, get_camera_names, load_object_poses, load_intrinsics, \
    get_num_frame
from renderer import create_scene, render_obj_pose, overlay_imgs, compare_gt_est


def pose_errors(p_est, p_gt, obj_id, models_info, error_types=['vsd', 'mssd', 'mspd'],
                K=None, depth_im=None, render=None):
    """
    Args:
        p_est, p_gt: 4x4
    """
    R_e, t_e = p_est[:3, :3], p_est[:3, 3:]
    R_g, t_g = p_gt[:3, :3], p_gt[:3, 3:]

    model_pts, model_sym, model_info_diameter = models_info[obj_id]['pts'], models_info[obj_id]['symmetry'], \
        models_info[obj_id]['diameter']

    spheres_overlap = np.linalg.norm(t_e - t_g) < model_info_diameter
    sphere_projections_overlap = misc.overlapping_sphere_projections(0.5 * model_info_diameter, t_e.squeeze(),
                                                                     t_g.squeeze())

    errors = {}
    for error_type in error_types:
        if error_type in ['mssd', 'add', 'adi'] and not spheres_overlap:
            errors[error_type] = float('inf')
            continue

        if error_type == 'vsd':
            # parameters
            vsd_deltas = 15
            vsd_taus = list(np.arange(0.05, 0.51, 0.05))
            vsd_normalized_by_diameter = True
            if not sphere_projections_overlap:
                e = [1.0] * len(vsd_taus)
            else:
                e = pose_error.vsd(R_e, t_e, R_g, t_g, depth_im, K, vsd_deltas, vsd_taus, vsd_normalized_by_diameter,
                                   model_info_diameter, render, obj_id, 'step')
        elif error_type == 'mssd':
            e = pose_error.mssd(R_e, t_e, R_g, t_g, model_pts, model_sym)
        elif error_type == 'mspd':
            e = pose_error.mspd(R_e, t_e, R_g, t_g, K, model_pts, model_sym)
        elif error_type == 'add':
            e = pose_error.add(R_e, t_e, R_g, t_g, model_pts)
        elif error_type == 'adi':
            e = pose_error.adi(R_e, t_e, R_g, t_g, model_pts)
        elif error_type == 'cus':
            if sphere_projections_overlap:
                e = pose_error.cus(R_e, t_e, R_g, t_g, K, render, obj_id)
            else:
                e = 1.0
        elif error_type == 'proj':
            e = pose_error.proj(R_e, t_e, R_g, t_g, K, model_pts)
        elif error_type == 'rete':
            e = pose_error.re(R_e, R_g), pose_error.te(t_e, t_g)
        elif error_type == 're':
            e = pose_error.re(R_e, R_g)
        elif error_type == 'te':
            e = pose_error.te(t_e, t_g)
        else:
            raise ValueError('Unknown pose error function.')

        # Normalize the errors by the object diameter.
        if error_type in ['ad', 'add', 'adi', 'mssd']:
            e /= model_info_diameter

        # Normalize the errors by the image width.
        if error_type in ['mspd']:
            e *= 640.0 / float(depth_im.shape[1])

        errors[error_type] = e

    return errors


def score_single_error(errors):
    scores = {}
    for error_type, e in errors.items():
        if error_type == 'vsd':
            counter = 0
            for t in np.arange(0.05, 0.51, 0.05):
                counter += np.count_nonzero(e < t)
            scores[error_type] = counter / 100.0
        elif error_type == 'mssd':
            counter = 0
            for t in np.arange(0.05, 0.51, 0.05):
                counter += 1 if e < t else 0
            scores[error_type] = counter / 10.0
        elif error_type == 'mspd':
            counter = 0
            for t in np.arange(5, 51, 5):
                counter += 1 if e < t else 0
            scores[error_type] = counter / 10.0
    average_recall = np.mean([s for s in scores.values()])
    return average_recall, scores


def single_image_score():
    scene_id = 48
    image_id = 1074
    ren_obj_id = 1
    est_pose_idx = 0
    render = True

    # test_path = '/media/gdk/Data/Datasets/dex_data/bop/s0/test/000001'
    test_path = f'/media/gdk/Hard_Disk/Datasets/bop_datasets/ycbv/test/{scene_id:06d}'

    color_im_path = f'{test_path}/rgb/{image_id:06d}.png'
    depth_im_path = f'{test_path}/depth/{image_id:06d}.png'
    color_im = cv2.imread(color_im_path)
    depth_im = load_depth(depth_im_path)

    intric_path = f'{test_path}/scene_camera.json'
    with open(intric_path) as f:
        intric = np.array(json.load(f)[str(image_id)]['cam_K']).reshape((3, 3))

    # load ground truth
    gt_path = f'{test_path}/scene_gt.json'
    gt_dict_id_poses = load_ground_truth(gt_path)[image_id]

    # load est pose
    bop_results_path = '/media/gdk/Hard_Disk/Datasets/bop_datasets/bop_output/challenge2020-223026_ycbv-test.csv'
    est_dict_id_poses = load_bop_est_pose(bop_results_path, scene_id, image_id)

    if ren_obj_id is not None:
        gt_dict_id_poses = {ren_obj_id: gt_dict_id_poses[ren_obj_id]}
        est_dict_id_poses = {ren_obj_id: est_dict_id_poses[ren_obj_id]}
    if est_pose_idx is not None:
        est_dict_id_poses[ren_obj_id] = [est_dict_id_poses[ren_obj_id][est_pose_idx]]

    # render gt and est pose
    if render:
        py_renderer = create_scene(intric, obj_model_paths.keys())

        gt_im = render_obj_pose(py_renderer, gt_dict_id_poses, unit='mm')
        # gt_im = overlay_imgs(color_im, gt_im)
        cv2.imshow('gt', gt_im)

        est_im = render_obj_pose(py_renderer, est_dict_id_poses, unit='mm')
        # est_im = overlay_imgs(color_im, est_im)
        cv2.imshow('est', est_im)

        # compare gt and est
        comp = compare_gt_est(gt_im, est_im)
        cv2.imshow('comp', comp)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # load models_info
    models_info = inout.load_json(models_info_path, keys_to_int=True)
    for obj_id in gt_dict_id_poses.keys():
        models_info[obj_id]['symmetry'] = misc.get_symmetry_transformations(models_info[obj_id], max_sym_disc_step=0.01)
        models_info[obj_id]['pts'] = inout.load_ply(ply_model_paths[obj_id])['pts']

    # create a renderer
    height, width = depth_im.shape
    ren = renderer.create_renderer(width, height, 'vispy', mode='depth')
    for obj_id in gt_dict_id_poses.keys():
        ren.add_object(obj_id, ply_model_paths[obj_id])

    for obj_id in gt_dict_id_poses.keys():
        for p_gt in gt_dict_id_poses[obj_id]:
            for p_est in est_dict_id_poses[obj_id]:
                errors = pose_errors(p_est, p_gt, obj_id, models_info, K=intric, depth_im=depth_im, render=ren)
                print(f"Obj_id: {obj_id}, Errors: {errors}")
                ar, score = score_single_error(errors)
                print(f"AR: {ar}, Scores: {score}")


def scene_score(scene_name: str, est_pose_file_paths: list, save_path, valid_objids=[12, 13]):
    """
    Calculate AR for the scene.
    files in est_pose_file_paths corresponds to camera_ids
    """
    scene_path = f"{dataset_path}/{scene_name}"
    camera_names = get_camera_names(scene_path)

    # find all obj_ids in scene
    gt_path = f"{scene_path}/{camera_names[0]}/scene_gt.json"
    gt_dict_frame_objid_poses = load_ground_truth(gt_path)
    obj_ids = set()
    for objid_poses in gt_dict_frame_objid_poses.values():
        obj_ids.update(list(objid_poses.keys()))
    print('obj_ids in ground truth:', obj_ids)

    # load models_info and renderer
    models_info = inout.load_json(models_info_path, keys_to_int=True)
    ren = renderer.create_renderer(resolution_width, resolution_height, 'vispy', mode='depth')
    for obj_id in obj_ids:
        if obj_id not in valid_objids:
            continue
        models_info[obj_id]['symmetry'] = misc.get_symmetry_transformations(models_info[obj_id], max_sym_disc_step=0.01)
        models_info[obj_id]['pts'] = inout.load_ply(ply_model_paths[obj_id])['pts']
        ren.add_object(obj_id, ply_model_paths[obj_id])

    arr_cameraid_frame_objid_AR = []
    for i, est_file_path in enumerate(est_pose_file_paths):
        camera_path = f"{scene_path}/{camera_names[i]}"

        # load ground truth
        gt_path = f"{camera_path}/scene_gt.json"
        gt_dict_frame_objid_poses = load_ground_truth(gt_path)

        # load est poses
        dict_obj_poses = load_object_poses(est_file_path)

        # load intric
        intric_path = f"{camera_path}/camera_meta.yml"
        intric = load_intrinsics(intric_path)

        # Loop through images
        for frame in range(get_num_frame(scene_path)):
            depth_im = load_depth(f'{camera_path}/depth/{frame:06d}.png')
            if frame not in gt_dict_frame_objid_poses:
                continue
            gt_objid_poses = gt_dict_frame_objid_poses[frame]

            for obj_id in gt_objid_poses.keys():
                if obj_id not in valid_objids:
                    continue
                if str(obj_id) not in dict_obj_poses:
                    arr_cameraid_frame_objid_AR.append([i, frame, int(obj_id), 0])
                else:
                    if dict_obj_poses[str(obj_id)][frame] is None:
                        arr_cameraid_frame_objid_AR.append([i, frame, int(obj_id), 0])
                    else:
                        p_est = dict_obj_poses[str(obj_id)][frame][0]
                        p_gt = gt_objid_poses[obj_id][0]
                        p_gt = np.r_[p_gt, [[0, 0, 0, 1]]]
                        errors = pose_errors(p_est, p_gt, obj_id, models_info, K=intric, depth_im=depth_im, render=ren)
                        ar, score = score_single_error(errors)
                        print([i, frame, int(obj_id), ar], errors)
                        arr_cameraid_frame_objid_AR.append([i, frame, int(obj_id), ar])

                        # render gt and est pose
                        if render:
                            py_renderer = create_scene(intric, obj_model_paths.keys())

                            gt_im = render_obj_pose(py_renderer, gt_dict_id_poses, unit='mm')
                            # gt_im = overlay_imgs(color_im, gt_im)
                            cv2.imshow('gt', gt_im)

                            est_im = render_obj_pose(py_renderer, est_dict_id_poses, unit='mm')
                            # est_im = overlay_imgs(color_im, est_im)
                            cv2.imshow('est', est_im)

                            # compare gt and est
                            comp = compare_gt_est(gt_im, est_im)
                            cv2.imshow('comp', comp)

                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

    arr_cameraid_frame_objid_AR = np.array(arr_cameraid_frame_objid_AR)
    arr_cameraid_frame_objid_AR.tofile(save_path, sep=',')


def print_scores(score_file_path, valid_objids=[12, 13]):
    arr_cameraid_frame_objid_AR = np.genfromtxt(score_file_path, delimiter=',').reshape((-1, 4))
    camera_ids = np.unique(arr_cameraid_frame_objid_AR[:, 0])
    obj_ids = np.unique(arr_cameraid_frame_objid_AR[:, 2])

    for c in camera_ids:
        text = f'camera {int(c)+1:02d}: '
        for obj_id in obj_ids:
            if obj_id not in valid_objids:
                continue
            ar = np.mean(arr_cameraid_frame_objid_AR[arr_cameraid_frame_objid_AR[:, 2] == obj_id, 3])
            text += f'obj {obj_id} AR={ar} '
        print(text)


if __name__ == '__main__':
    # single_image_score()
    scene_name = 'scene_2210232307_01'
    est_pose_file_paths = []
    scene_path = f"{dataset_path}/{scene_name}"
    camera_names = get_camera_names(scene_path)
    for camera_name in camera_names:
        camera_path = f"{scene_path}/{camera_name}"
        est_pose_file_paths.append(f'{camera_path}/object_pose/PoseRBPF/frame_obj_poses.json')
    scene_score(scene_name, est_pose_file_paths, f'{scene_path}/object_pose/PoseRBPF/scores.csv')
    print_scores(f'{scene_path}/object_pose/PoseRBPF/scores.csv')



















