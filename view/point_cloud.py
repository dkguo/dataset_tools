"""
Manual annotation tool with multiple views

Using RGB, Depth and Models the tool will generate the "scene_gt.json" annotation file

Other annotations can be generated usign other scripts [calc_gt_info.py, calc_gt_masks.py, ....]

original repo: https://github.com/FLW-TUDO/3d_annotation_tool

"""
import argparse
import json
import os

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from dataset_tools.dataset_config import dataset_path
from dataset_tools.loaders import load_intrinsics, get_depth_scale, get_camera_names, load_extrinsics
from dataset_tools.renderer import model_names, ply_model_paths


class AnnotationScene:
    def __init__(self, scene_point_cloud, image_num):
        self.annotation_scene = scene_point_cloud
        self.image_num = image_num

        self.obj_list = list()

    def add_obj(self, obj_geometry, obj_name, obj_instance, transform=np.identity(4)):
        self.obj_list.append(self.SceneObject(obj_geometry, obj_name, obj_instance, transform))

    def get_objects(self):
        return self.obj_list[:]

    class SceneObject:
        def __init__(self, obj_geometry, obj_name, obj_instance, transform):
            self.obj_geometry = obj_geometry
            self.obj_name = obj_name
            self.obj_instance = obj_instance
            self.transform = transform


class Settings:
    UNLIT = "defaultUnlit"

    def __init__(self):
        self.bg_color = gui.Color(1, 1, 1)
        self.show_axes = False
        self.highlight_obj = True

        self.apply_material = True  # clear to False after processing

        self.scene_material = rendering.MaterialRecord()
        self.scene_material.base_color = [0.9, 0.9, 0.9, 1.0]
        self.scene_material.shader = Settings.UNLIT

        self.annotation_obj_material = rendering.MaterialRecord()
        self.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
        self.annotation_obj_material.shader = Settings.UNLIT


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    MATERIAL_NAMES = ["Unlit"]
    MATERIAL_SHADERS = [
        Settings.UNLIT
    ]

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self.pc_scene_widget.scene.set_background(bg_color)

        if self.settings.apply_material:
            self.pc_scene_widget.scene.modify_geometry_material("annotation_scene", self.settings.scene_material)
            self.settings.apply_material = False

        self._highlight_obj.checked = self.settings.highlight_obj
        self._point_size.double_value = self.settings.scene_material.point_size

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)
        self.pc_scene_widget.frame = gui.Rect(0, r.y, r.get_right() - width, r.height)

    def __init__(self, width, height, scene_path):
        self.active_camera_view = 0
        self.pre_load_meshes = {}
        self.camera_names = sorted(get_camera_names(scene_path))
        self.extrinsics = load_extrinsics(f'{scene_path}/extrinsics.yml')
        self.rgb_imgs = []
        self.active_objs_pose = {}
        self.scene_path = scene_path
        self.settings = Settings()

        # 3D widget
        self.window = gui.Application.instance.create_window("View Point Cloud", width, height)
        w = self.window  # to make the code more concise

        self.pc_scene_widget = gui.SceneWidget()
        self.pc_scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.pc_scene_widget)

        # ---- Settings panel ----
        em = w.theme.font_size

        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        view_ctrls = gui.CollapsableVert("View control", 0, gui.Margins(em, 0, 0, 0))
        view_ctrls.set_is_open(True)

        self._highlight_obj = gui.Checkbox("Highligh annotation objects")
        self._highlight_obj.set_on_checked(self._on_highlight_obj)
        view_ctrls.add_child(self._highlight_obj)

        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 5)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        view_ctrls.add_child(grid)

        self._settings_panel.add_child(view_ctrls)
        # ----

        w.set_on_layout(self._on_layout)
        w.add_child(self._settings_panel)

        # 3D Annotation tool options
        annotation_objects = gui.CollapsableVert("Annotation Objects", 0.33 * em, gui.Margins(em, 0, 0, 0))
        annotation_objects.set_is_open(True)
        self._meshes_available = gui.ListView()
        self._meshes_used = gui.ListView()

        self._scene_control = gui.CollapsableVert("Scene Control", 0.33 * em, gui.Margins(em, 0, 0, 0))
        self._scene_control.set_is_open(True)

        self._images_buttons_label = gui.Label("Images:")

        self._pre_image_button = gui.Button("Previous Image")
        self._pre_image_button.horizontal_padding_em = 0.8
        self._pre_image_button.vertical_padding_em = 0
        self._pre_image_button.set_on_clicked(self._on_previous_image)
        self._next_image_button = gui.Button("Next Image")
        self._next_image_button.horizontal_padding_em = 0.8
        self._next_image_button.vertical_padding_em = 0
        self._next_image_button.set_on_clicked(self._on_next_image)

        # 2 rows for sample and scene control
        h = gui.Horiz(0.4 * em)  # row 2
        h.add_stretch()
        self._scene_control.add_child(h)
        self._view_numbers = gui.Horiz(0.4 * em)
        self._image_number = gui.Label("Image: " + f'{0:06}')
        self._view_numbers.add_child(self._image_number)
        self._scene_control.add_child(self._view_numbers)

        h = gui.Horiz(0.4 * em)  # row 1
        h.add_stretch()
        # h.add_child(self._images_buttons_label)
        h.add_child(self._pre_image_button)
        h.add_child(self._next_image_button)
        h.add_stretch()
        self._scene_control.add_child(h)

        self._settings_panel.add_child(self._scene_control)

        # ---- Menu ----
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        # ---- annotation tool settings ----
        self._on_point_size(1)  # set default size to 1

        self._apply_settings()

        self._annotation_scene = None

    def _update_img_numbers(self):
        self._image_number.text = "Image: " + f'{self._annotation_scene.image_num:06}'

    def _on_highlight_obj(self, light):
        self.settings.highlight_obj = light
        if light:
            self.settings.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
        elif not light:
            self.settings.annotation_obj_material.base_color = [0.9, 0.9, 0.9, 1.0]

        self._apply_settings()

        # update current object visualization
        meshes = self._annotation_scene.get_objects()
        for mesh in meshes:
            self.pc_scene_widget.scene.modify_geometry_material(mesh.obj_name, self.settings.annotation_obj_material)

    def _on_point_size(self, size):
        self.settings.scene_material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("BOP manual annotation tool"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def _obj_instance_count(self, mesh_to_add, meshes):
        types = [i[:-2] for i in meshes]  # remove last 3 character as they present instance number (OBJ_INSTANCE)
        equal_values = [i for i in range(len(types)) if types[i] == mesh_to_add]
        count = 0
        if len(equal_values):
            indices = np.array(meshes)
            indices = indices[equal_values]
            indices = [int(x[-1]) for x in indices]
            count = max(indices) + 1
        return count

    def _make_point_cloud(self, rgb_img, depth_img, cam_K):
        # convert images to open3d types
        rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        depth_img_o3d = o3d.geometry.Image(depth_img)

        # convert image to point cloud
        intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_img.shape[0], rgb_img.shape[1],
                                                      cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d,
                                                                  depth_scale=1, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        return pcd

    def scene_load(self, image_num):
        self._annotation_changed = False

        self.pc_scene_widget.scene.clear_geometry()
        geometry = None

        cam_K = load_intrinsics(f'{self.scene_path}/{self.camera_names[0]}/camera_meta.yml')
        depth_scale = get_depth_scale(f'{self.scene_path}/{self.camera_names[0]}/camera_meta.yml', convert2unit='m')

        rgb_path = os.path.join(self.scene_path, f'{self.camera_names[0]}', 'rgb', f'{image_num:06}' + '.png')
        rgb_img = cv2.imread(rgb_path)
        depth_path = os.path.join(self.scene_path, f'{self.camera_names[0]}', 'depth', f'{image_num:06}' + '.png')
        depth_img = cv2.imread(depth_path, -1)
        depth_img = np.float32(depth_img * depth_scale)

        self.rgb_imgs = []
        for camera in self.camera_names:
            rgb_path = os.path.join(self.scene_path, camera, 'rgb', f'{image_num:06}' + '.png')
            self.rgb_imgs.append(cv2.imread(rgb_path))

        try:
            geometry = self._make_point_cloud(rgb_img, depth_img, cam_K)
        except Exception:
            print("Failed to load scene.")

        if geometry is not None:
            print("[Info] Successfully read scene ", image_num)
            if not geometry.has_normals():
                geometry.estimate_normals()
            geometry.normalize_normals()
        else:
            print("[WARNING] Failed to read points")

        try:
            self.pc_scene_widget.scene.add_geometry("annotation_scene", geometry, self.settings.scene_material,
                                                    add_downsampled_copy_for_fast_rendering=True)
            bounds = geometry.get_axis_aligned_bounding_box()
            self.pc_scene_widget.setup_camera(60, bounds, bounds.get_center())
            center = np.array([0, 0, 0])
            eye = center + np.array([0, 0, -0.5])
            up = np.array([0, -1, 0])
            self.pc_scene_widget.look_at(center, eye, up)

            self._annotation_scene = AnnotationScene(geometry, image_num)
            self._meshes_used.set_items([])  # clear list from last loaded scene

            # load values if an annotation already exists
            self._load_annotation(image_num)

        except Exception as e:
            print(e)

        self._update_img_numbers()

    def _load_annotation(self, image_num):
        scene_gt_path = os.path.join(self.scene_path, f'{self.camera_names[0]}', 'scene_gt.json')
        with open(scene_gt_path) as scene_gt_file:
            data = json.load(scene_gt_file)
            if str(image_num) not in data.keys():
                print(f'No gt in image {image_num}')
                return
            scene_data = data[str(image_num)]
            for obj in scene_data:
                # add object to annotation_scene object
                obj_geometry = o3d.io.read_point_cloud(ply_model_paths[int(obj['obj_id'])])
                obj_geometry.points = o3d.utility.Vector3dVector(
                    np.array(obj_geometry.points) / 1000)  # convert mm to meter
                model_name = model_names[int(obj['obj_id']) - 1]
                meshes = self._annotation_scene.get_objects()  # update list after adding current object
                meshes = [i.obj_name for i in meshes]
                obj_instance = self._obj_instance_count(model_name, meshes)
                obj_name = model_name + '_' + str(obj_instance)
                translation = np.array(np.array(obj['cam_t_m2c']), dtype=np.float64) / 1000  # convert to meter
                orientation = np.array(np.array(obj['cam_R_m2c']), dtype=np.float64)
                transform = np.concatenate((orientation.reshape((3, 3)), translation.reshape(3, 1)), axis=1)
                transform_cam_to_obj = np.concatenate(
                    (transform, np.array([0, 0, 0, 1]).reshape(1, 4)))  # homogeneous transform

                self._annotation_scene.add_obj(obj_geometry, obj_name, obj_instance, transform_cam_to_obj)
                # adding object to the scene
                obj_geometry.translate(transform_cam_to_obj[0:3, 3])
                center = obj_geometry.get_center()
                obj_geometry.rotate(transform_cam_to_obj[0:3, 0:3], center=center)
                self.pc_scene_widget.scene.add_geometry(obj_name, obj_geometry, self.settings.annotation_obj_material,
                                                        add_downsampled_copy_for_fast_rendering=True)
                # active_meshes.append(obj_name)
                self.active_objs_pose[obj_name] = (int(obj['obj_id']), transform_cam_to_obj)

        meshes = self._annotation_scene.get_objects()  # update list after adding current object
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)

    def update_obj_list(self):
        self._meshes_available.set_items(model_names)

    def _on_next_image(self):
        if self._annotation_scene.image_num + 1 >= len(
                next(os.walk(os.path.join(self.scene_path, f'{self.camera_names[0]}', 'depth')))[
                    2]):  # 2 for files which here are the how many depth images
            self._on_error("There is no next image.")
            return
        self.scene_load(self._annotation_scene.image_num + 1)

    def _on_previous_image(self):
        if self._annotation_scene.image_num - 1 < 0:
            self._on_error("There is no image number before image 0.")
            return
        self.scene_load(self._annotation_scene.image_num - 1)

    def _on_error(self, err_msg):
        dlg = gui.Dialog("Error")

        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(err_msg))

        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)


def main():
    parser = argparse.ArgumentParser(description='Annotation tool.')
    parser.add_argument('--scene_name', default='scene_2210232307_01')
    parser.add_argument('--start_frame', type=int, default=20)

    args = parser.parse_args()
    scene_name = args.scene_name
    scene_path = f'{dataset_path}/{scene_name}'
    start_image_num = args.start_frame

    gui.Application.instance.initialize()
    w = AppWindow(2048, 1536, scene_path)

    w.scene_load(start_image_num)
    w.update_obj_list()

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
