import os

import numpy as np
import pandas as pd
import pybullet_planning as pp

from dataset_tools.config import dataset_path
from dataset_tools.utils.camera_parameter import load_extrinsics
from dataset_tools.utils.name import get_num_frame, get_camera_names, get_newest_scene_name


class Table:
    def __init__(self, table_path=None, scene_name=None, **kwargs):
        self.table_path = table_path
        self.scene_name = scene_name
        if table_path is None:
            self.table_path = self.get_default_table_path()
        if os.path.exists(self.table_path):
            self.table = self.load(self.table_path, **kwargs)
        else:
            self.table = self.create_empty_table()

    def create_empty_table(self):
        pass

    def get_default_table_path(self):
        pass

    def load(self, table_path=None, **kwargs):
        if table_path is None:
            table_path = self.table_path
        df = pd.read_csv(table_path)
        self.table = df.to_records(index=False)
        return self.table

    def save(self, table_path=None, tolist_cols=None, sort_by=None):
        if tolist_cols is None:
            tolist_cols = []
        if table_path is None:
            table_path = self.table_path
        if not os.path.exists(os.path.dirname(table_path)):
            os.makedirs(os.path.dirname(table_path))
        df = pd.DataFrame.from_records(self.table)
        for col in tolist_cols:
            df[col] = df[col].apply(np.ndarray.tolist)
        if sort_by is not None:
            df = df.sort_values(by=sort_by)
        df.to_csv(table_path, index=False)
        print(f'Saved table to {table_path}')


class ObjectPoseTable(Table):
    def __init__(self, opt_path=None, scene_name=None, **kwargs):
        super().__init__(opt_path, scene_name, **kwargs)

    def create_empty_table(self):
        return np.empty(0, dtype=[
            ('scene_name', 'U20'),
            ('camera_name', 'U20'),
            ('frame', 'i4'),
            ('object_name', 'U20'),
            ('predictor', 'U20'),
            ('pose', 'O')
        ])

    def get_default_table_path(self):
        return None if self.scene_name is None else f'{dataset_path}/{self.scene_name}/object_pose_table.csv'

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

    def save(self, table_path=None, tolist_cols=None, sort_by=None):
        if tolist_cols is None:
            tolist_cols = ['pose']
        super().save(table_path, tolist_cols)

    def update(self, object_name, pose: np.ndarray,
               scene_name='undefinded', camera_name='undefinded', frame=-1, predictor='undefinded'):
        mask = self._create_mask(object_name, scene_name, camera_name, frame, predictor)
        if len(self.table[mask]) == 0:
            self.table = np.append(self.table,
                                   np.array([(scene_name, camera_name, frame, object_name, predictor, pose)],
                                            dtype=self.table.dtype))
        else:
            self.table[np.nonzero(mask)[0][0]]['pose'] = pose
        return self.table

    def lookup(self, object_name,
               scene_name='any', camera_name='any', frame=-1, predictor='any'):
        mask = self._create_mask(object_name, scene_name, camera_name, frame, predictor)
        return self.table[mask]['pose']

    def _create_mask(self, object_name,
                     scene_name='any', camera_name='any', frame=-1, predictor='any'):
        mask = self.table['object_name'] == object_name
        if scene_name != 'any':
            mask = np.logical_and(mask, self.table['scene_name'] == scene_name)
        if camera_name != 'any':
            mask = np.logical_and(mask, self.table['camera_name'] == camera_name)
        if frame != -1:
            mask = np.logical_and(mask, self.table['frame'] == frame)
        if predictor != 'any':
            mask = np.logical_and(mask, self.table['predictor'] == predictor)
        return mask

    def lookup_trajectory(self, start_frame, end_frame, object_name,
                          scene_name='any', camera_name='any', predictor='any'):
        self.table.sort(order=['frame'])
        mask = self._create_mask(object_name, scene_name, camera_name, predictor=predictor)
        mask = np.logical_and.reduce([mask, self.table['frame'] >= start_frame, self.table['frame'] <= end_frame])
        traj = self.table[mask]['pose']
        frames = self.table[mask]['frame']
        durations = np.r_[0, np.diff(frames)] / 30.0
        return traj, durations


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


def mats2qts(mats):
    qts = []
    for mat in mats:
        qts.append(pp.pose_from_tform(mat))
    return qts
