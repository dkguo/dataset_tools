from dataset_tools.view.renderer import create_renderer
create_renderer()


from open3d.visualization import gui

from dataset_tools.loaders import load_intrinsics
from dataset_tools.view.open3d_window import Open3dWindow


class Annotation(Open3dWindow):
    def __init__(self, scene_name, init_frame_num, width=640*3+408, height=480*3, hand_mask_dir=None, obj_pose_file=None):
        super().__init__(width, height, scene_name, 'Annotation')
        self.renderers = []
        # for camera in self.camera_names[0]:
        # cam_K = load_intrinsics(f'{self.scene_path}/{self.camera_names[0]}/camera_meta.yml')
        # self.renderers.append(create_renderer(cam_K))


if __name__ == "__main__":
    scene_name = 'scene_2210232307_01'
    start_image_num = 0
    hand_mask_dir = 'hand_pose/d2/mask'
    obj_pose_file = 'object_pose/multiview_medium/object_poses.csv'


    gui.Application.instance.initialize()
    w = Annotation(scene_name, start_image_num, obj_pose_file=obj_pose_file)

    gui.Application.instance.run()