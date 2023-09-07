import os

import numpy as np
import pandas as pd

from dataset_tools.config import dataset_path
from dataset_tools.utils.camera_parameter import load_extrinsics
from dataset_tools.utils.name import get_num_frame, get_camera_names, get_scene_name_from_path


class Table:
    def __init__(self, table_path=None, scene_name=None):
        self.table_path = table_path
        self.scene_name = scene_name
        if table_path is None:
            self.table = self.create_empty_table()
        else:
            self.table = self.load(table_path)

    def create_empty_table(self):
        pass

    def load(self, table_path=None):
        if table_path is None:
            table_path = self.table_path
        df = pd.read_csv(table_path)
        self.table = df.to_records(index=False)
        return self.table

    def save(self, table_path=None, tolist_cols=[]):
        if not os.path.exists(os.path.dirname(table_path)):
            os.makedirs(os.path.dirname(table_path))
        df = pd.DataFrame.from_records(self.table)
        for col in tolist_cols:
            df[col] = df[col].apply(np.ndarray.tolist)
        df.to_csv(table_path, index=False)

    def append(self, *args, **kwargs):
        pass


class ObjectPoseTable(Table):
    def __init__(self, opt_path=None, scene_name=None):
        super().__init__(opt_path, scene_name)

    def create_empty_table(self):
        return np.empty(0, dtype=[
            ('scene_name', 'U20'),
            ('camera_name', 'U20'),
            ('frame', 'i4'),
            ('object_name', 'U20'),
            ('predictor', 'U20'),
            ('pose', 'O')
        ])

    def load(self, table_path=None, only_valid_pose=False, fill_nan=False, mm2m=False):
        df = pd.read_csv(table_path, converters={'pose': lambda x: eval(f'np.array({x})')})
        df = df.sort_values(by=['frame', 'object_name'])
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
            object_names = set(opt['object_name'])
            scene_name = opt[0]['scene_name']
            for object_name in object_names:
                opt_obj = opt[opt['object_name'] == object_name]
                for frame in range(get_num_frame(f'{dataset_path}/{scene_name}')):
                    if frame not in opt_obj['frame']:
                        opt = np.append(opt, opt[-1])
                        opt[-1]['object_name'] = object_name
                        opt[-1]['frame'] = frame
                        opt[-1]['pose'] = np.full((4, 4), np.nan)

        self.table = opt
        return opt

    def save(self, table_path=None, tolist_cols=['pose']):
        super().save(table_path, tolist_cols)

    def update(self, object_name, pose: np.ndarray,
                    scene_name='undefinded', camera_name='undefinded', frame=-1, predictor='undefinded'):
        mask = create_mask(self.table, object_name, scene_name, camera_name, frame, predictor)
        if len(self.table[mask]) == 0:
            self.table = np.append(self.table, np.array([(scene_name, camera_name, frame, object_name, predictor, pose)],
                                          dtype=self.table.dtype))
        else:
            self.table[np.nonzero(mask)[0][0]]['pose'] = pose
        return self.table


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


def get_opt_path(scene_name):
    return f'{dataset_path}/{scene_name}/object_pose_table.csv'


def record_pose(opt_path_or_scene_name,
                object_name, pose: np.ndarray, scene_name='undefinded', camera_name='undefinded', frame=-1,
                predictor='undefinded'):
    opt_path = opt_path_or_scene_name if '.csv' in opt_path_or_scene_name else get_opt_path(opt_path_or_scene_name)
    scene_name = get_scene_name_from_path(opt_path) if scene_name == 'undefinded' else scene_name
    opt = load_object_pose_table(opt_path) if os.path.exists(opt_path) else create_empty_opt()
    opt = update_pose(opt, object_name, pose, scene_name, camera_name, frame, predictor)
    save_object_pose_table(opt, opt_path)



def load_object_pose(opt_or_opt_path_or_scene_name,
                     object_name, scene_name='undefinded', camera_name='undefinded', frame=-1, predictor='undefinded'):
    if isinstance(opt_or_opt_path_or_scene_name, np.recarray):
        opt = opt_or_opt_path_or_scene_name
    else:
        opt_path = opt_or_opt_path_or_scene_name if '.csv' in opt_or_opt_path_or_scene_name else get_opt_path(
            opt_or_opt_path_or_scene_name)
        if os.path.exists(opt_path):
            opt = load_object_pose_table(opt_path)
            scene_name = get_scene_name_from_path(opt_path) if scene_name == 'undefinded' else scene_name
        else:
            print(f'{opt_path} does not exist')
            return None
    mask = create_mask(opt, object_name, scene_name, camera_name, frame, predictor)
    return None if len(opt[mask]) == 0 else opt[mask]['pose'][0]


def create_mask(opt, object_name, scene_name='undefinded', camera_name='undefinded', frame=-1, predictor='undefinded'):
    mask = opt['object_name'] == object_name
    for query in [scene_name, camera_name, frame, predictor]:
        if query != 'undefinded' and query != -1:
            mask = np.logical_and(mask, opt['scene_name'] == query)
    return mask
