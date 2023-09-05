import json

import cv2
import numpy as np
from numpy.lib.recfunctions import append_fields
from tqdm import tqdm

from config import obj_ply_paths, models_info_path, obj_model_paths, dataset_path, resolution_width, resolution_height
from dataset_tools.bop_toolkit.bop_toolkit_lib import pose_error, misc, inout, renderer
from dataset_tools.bop_toolkit.bop_toolkit_lib.inout import load_depth
from utils import get_camera_names, load_intrinsics, save_object_pose_table, load_object_pose_table
from dataset_tools.view.renderer import create_renderer, render_obj_pose#, compare_gt_est


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


def score_single_image():
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
        py_renderer = create_renderer(intric, obj_model_paths.keys())

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
        models_info[obj_id]['pts'] = inout.load_ply(obj_ply_paths[obj_id])['pts']

    # create a renderer
    height, width = depth_im.shape
    ren = renderer.create_renderer(width, height, 'vispy', mode='depth')
    for obj_id in gt_dict_id_poses.keys():
        ren.add_object(obj_id, obj_ply_paths[obj_id])

    for obj_id in gt_dict_id_poses.keys():
        for p_gt in gt_dict_id_poses[obj_id]:
            for p_est in est_dict_id_poses[obj_id]:
                errors = pose_errors(p_est, p_gt, obj_id, models_info, K=intric, depth_im=depth_im, render=ren)
                print(f"Obj_id: {obj_id}, Errors: {errors}")
                ar, score = score_single_error(errors)
                print(f"AR: {ar}, Scores: {score}")


def score_scene(scene_name, est_pose_file_paths, render=False):
    """
    Calculate AR for the scene.
    files in est_pose_file_paths corresponds to camera_ids
    AR are stored in average_recall column of est_pose_file_paths
    """
    scene_path = f"{dataset_path}/{scene_name}"
    camera_names = get_camera_names(scene_path)

    # find all obj_ids in scene_gt
    gt_path = f"{scene_path}/{camera_names[0]}/object_pose/ground_truth.csv"
    gt_opt = load_object_pose_table(gt_path)
    obj_ids = set(gt_opt['obj_id'])
    print('obj_ids in ground truth:', obj_ids)

    # load models_info and renderer
    models_info = inout.load_json(models_info_path, keys_to_int=True)
    ren = renderer.create_renderer(resolution_width, resolution_height, 'vispy', mode='depth')
    for obj_id in obj_ids:
        if obj_id in models_info:
            models_info[obj_id]['symmetry'] = misc.get_symmetry_transformations(models_info[obj_id], max_sym_disc_step=0.01)
            models_info[obj_id]['pts'] = inout.load_ply(obj_ply_paths[obj_id])['pts']
            ren.add_object(obj_id, obj_ply_paths[obj_id])

    # loop through cameras
    for camera_i, est_file_path in enumerate(est_pose_file_paths):
        print(camera_names[camera_i])
        camera_path = f"{scene_path}/{camera_names[camera_i]}"

        gt_opt = load_object_pose_table(f"{camera_path}/object_pose/ground_truth.csv")
        est_opt = load_object_pose_table(est_file_path)
        intric = load_intrinsics(f"{camera_path}/camera_meta.yml")

        if 'average_recall' not in est_opt.dtype.names:
            est_opt = append_fields(est_opt, 'average_recall', np.zeros(est_opt.shape[0]), dtypes='f', usemask=False)
        else:
            est_opt['average_recall'] = 0.0

        # Loop through gt pose
        for frame, obj_id, p_gt in tqdm(gt_opt[['frame', 'obj_id', 'pose']]):
            if not np.logical_and(est_opt['frame'] == frame, est_opt['obj_id'] == obj_id).any():
                # pose is not predicted, add a dummy pose
                est_opt = np.append(est_opt, [est_opt[-1]])
                est_opt[['frame', 'obj_id', 'pose', 'average_recall']][-1] = (frame, obj_id, np.zeros(1), 0)
            else:
                # loop through est_pose matching frame and obj_id
                depth_im = load_depth(f'{camera_path}/depth/{frame:06d}.png')
                mask = np.logical_and(est_opt['frame'] == frame, est_opt['obj_id'] == obj_id)
                for k in mask.nonzero()[0]:
                    p_est = est_opt['pose'][k]

                    # dummy pose
                    if len(p_est) != 4:
                        continue

                    # render gt and est pose
                    if render:
                        print(f'Frame: {frame}, obj_id: {obj_id}')
                        py_renderer = create_renderer(intric, obj_ids)

                        gt_im = render_obj_pose(py_renderer, list_id_pose=[(obj_id, p_gt)], unit='mm')
                        # gt_im = overlay_imgs(color_im, gt_im)
                        cv2.imshow('gt', gt_im)

                        est_im = render_obj_pose(py_renderer, list_id_pose=[(obj_id, p_est)], unit='mm')
                        # est_im = overlay_imgs(color_im, est_im)
                        cv2.imshow('est', est_im)

                        # compare gt and est
                        comp = compare_gt_est(gt_im, est_im)
                        cv2.imshow('comp', comp)

                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    errors = pose_errors(p_est, p_gt, obj_id, models_info, K=intric, depth_im=depth_im, render=ren,
                                         error_types=['vsd', 'mssd', 'mspd'])
                    ar, score = score_single_error(errors)
                    if ar > est_opt[k]['average_recall']:
                        est_opt['average_recall'][k] = ar

                    # print(f'Frame: {frame}, obj_id: {obj_id}, AR: {ar}')

        # save
        save_object_pose_table(est_opt, est_file_path)


def print_scores(est_pose_file_paths, test_obj_ids=[]):
    obj_ars = {}
    all_ars = np.zeros(len(est_pose_file_paths))
    test_ars = np.zeros(len(est_pose_file_paths))
    for camera_i, file_path in enumerate(est_pose_file_paths):
        est_opt = load_object_pose_table(file_path)
        obj_ids = set(est_opt['obj_id'])
        for obj_id in obj_ids:
            ar = np.mean(est_opt[est_opt['obj_id'] == obj_id]['average_recall'])
            if obj_id not in obj_ars:
                obj_ars[obj_id] = np.zeros(len(est_pose_file_paths))
            obj_ars[obj_id][camera_i] = ar

        if len(test_obj_ids) > 0:
            ar = np.mean(est_opt[np.isin(est_opt['obj_id'], test_obj_ids)]['average_recall'])
            test_ars[camera_i] = ar
        ar = np.mean(est_opt['average_recall'])
        all_ars[camera_i] = ar

    print('         Camera 01     02     03     04     05     06     07     08      Max     Avg')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    for obj_id, ars in sorted(obj_ars.items()):
        print(f'Obj {obj_id}\t\t{ars}\t{np.max(ars):0.3f}\t{np.mean(ars):0.3f}')
    print('--')
    print(f'Test Objs\t{test_ars}\t{np.max(test_ars):0.3f}\t{np.mean(test_ars):0.3f}')
    print(f'All Objs\t{all_ars}\t{np.max(all_ars):0.3f}\t{np.mean(all_ars):0.3f}')


if __name__ == '__main__':
    scene_name = 'scene_2210232307_01'
    predictor = 'deepim'
    scene_path = f"{dataset_path}/{scene_name}"

    est_pose_file_paths = []
    for camera_name in get_camera_names(scene_path):
        camera_path = f"{scene_path}/{camera_name}"
        est_pose_file_paths.append(f'{camera_path}/object_pose/{predictor}/object_poses.csv')

    score_scene(scene_name, est_pose_file_paths, render=False)
    print_scores(est_pose_file_paths, test_obj_ids=[21, 24])



















