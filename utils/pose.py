def load_object_pose_table(file_path, only_valid_pose=False, fill_nan=False, mm2m=False):
    """
    Returns:
        obj_pose_table (opt), numpy recarray
    """
    df = pd.read_csv(file_path, converters={'pose': lambda x: eval(f'np.array({x})')})
    df = df.sort_values(by=['frame', 'obj_id'])
    if only_valid_pose:
        drop_idxs = []
        for i, pose in enumerate(df['pose']):
            if len(pose) != 4:
                drop_idxs.append(i)
        df = df.drop(index=drop_idxs)
    opt = df.to_records(index=False)

    if mm2m:
        for i in range(len(opt)):
            opt[i]['pose'][:3, 3] /= 1000.0

    if fill_nan:
        obj_ids = set(opt['obj_id'])
        scene_name = opt[0]['scene_name']
        for obj_id in obj_ids:
            opt_obj = opt[opt['obj_id'] == obj_id]
            for frame in range(get_num_frame(f'{dataset_path}/{scene_name}')):
                if frame not in opt_obj['frame']:
                    opt = np.append(opt, opt[-1])
                    opt[-1]['obj_id'] = obj_id
                    opt[-1]['frame'] = frame
                    opt[-1]['pose'] = np.full((4, 4), np.nan)
    return opt


def create_empty_opt():
    return np.empty(0, dtype=[('scene_name', 'U20'),
                              ('camera_name', 'U20'),
                              ('frame', 'i4'),
                              ('obj_id', 'i4'),
                              ('predictor', 'U20'),
                              ('pose', 'O')])


def save_object_pose_table(opt, file_path, col_tolist='pose'):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    df = pd.DataFrame.from_records(opt)
    df[col_tolist] = df[col_tolist].apply(np.ndarray.tolist)
    df.to_csv(file_path, index=False)


def load_all_opts(scene_path, opt_file_name, convert2origin=False):
    extrinsics = load_extrinsics(f'{scene_path}/extrinsics.yml', to_mm=True)
    opt_all = []
    for camera_name in get_camera_names(scene_path):
        opt_path = f"{scene_path}/{camera_name}/{opt_file_name}"
        opt = load_object_pose_table(opt_path, only_valid_pose=True)
        if convert2origin:
            origin_camera = extrinsics[camera_name]
            opt['pose'] = [origin_camera @ p for p in opt['pose']]
        opt_all.append(opt)
    opt_all = np.hstack(opt_all)
    return opt_all


def load_infra_pose(scene_name, infra_name='sink_unit'):
    ipt = load_object_pose_table(f"{dataset_path}/{scene_name}/infra_poses.csv", only_valid_pose=True)
    infra_pose = ipt[ipt['name'] == infra_name]['pose'][0]
    return infra_pose


def load_table_to_array(file_path):
    df = pd.read_csv(file_path)
    pt = df.to_records(index=False)
    return pt
