"""
Manual annotation tool with multiple views

Using RGB, Depth and Models the tool will generate the "scene_gt.json" annotation file

Other annotations can be generated usign other scripts [calc_gt_info.py, calc_gt_masks.py, ....]

original repo: https://github.com/FLW-TUDO/3d_annotation_tool

"""
import time

# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import argparse

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import json
import cv2

from config import dataset_path
from loaders import load_intrinsics, get_depth_scale, get_camera_names, load_extrinsics
from process.helpers import add_border, collage_imgs
from dataset_tools.view.renderer import create_scene, model_names, render_obj_pose, overlay_imgs, ply_model_paths

dist = 0.05
deg = 30


class AnnotationScene:
    def __init__(self, scene_point_cloud, image_num):
        self.annotation_scene = scene_point_cloud
        self.image_num = image_num

        self.obj_list = list()

    def add_obj(self, obj_geometry, obj_name, obj_instance, transform=np.identity(4)):
        self.obj_list.append(self.SceneObject(obj_geometry, obj_name, obj_instance, transform))

    def get_objects(self):
        return self.obj_list[:]

    def remove_obj(self, index):
        self.obj_list.pop(index)

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
        self.pc_scene_widget.scene.show_axes(self.settings.show_axes)

        if self.settings.apply_material:
            self.pc_scene_widget.scene.modify_geometry_material("annotation_scene", self.settings.scene_material)
            self.settings.apply_material = False

        self._show_axes.checked = self.settings.show_axes
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
        self.im_scene_widget.frame = gui.Rect(0, r.y, r.get_right() - width, height)

    def __init__(self, width, height, scene_path):
        self.active_camera_view = 0
        self.pre_load_meshes = {}
        self.camera_names = sorted(get_camera_names(scene_path))
        self.py_renderers = []
        for camera in self.camera_names:
            cam_K = load_intrinsics(f'{scene_path}/{camera}/camera_meta.yml')
            self.py_renderers.append(create_scene(cam_K, pre_load_meshes=self.pre_load_meshes))
        self.extrinsics = load_extrinsics(f'{scene_path}/extrinsics.yml')
        self.rgb_imgs = []
        self.active_objs_pose = {}
        self.scene_path = scene_path
        self.settings = Settings()

        # 3D widget
        self.pt_window = gui.Application.instance.create_window("Point Cloulds", 640, 480)
        self.pc_scene_widget = gui.SceneWidget()
        self.pc_scene_widget.scene = rendering.Open3DScene(self.pt_window.renderer)
        self.pt_window.add_child(self.pc_scene_widget)

        # Window to display redenered image
        self.window = gui.Application.instance.create_window("BOP manual annotation tool", width, height)
        w = self.window  # to make the code more concise

        self.im_scene_widget = gui.SceneWidget()
        self.window.add_child(self.im_scene_widget)
        self.im_scene_widget.scene = rendering.Open3DScene(self.window.renderer)

        im = np.ones((480, 640, 3)).astype(np.uint8)
        im = o3d.geometry.Image(im)
        self.im_scene_widget.scene.set_background([0, 0, 0, 0], im)

        # ---- Settings panel ----
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        view_ctrls = gui.CollapsableVert("View control", 0, gui.Margins(em, 0, 0, 0))
        view_ctrls.set_is_open(True)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)

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
        annotation_objects = gui.CollapsableVert("Annotation Objects", 0.33 * em,
                                                 gui.Margins(em, 0, 0, 0))
        annotation_objects.set_is_open(True)
        self._meshes_available = gui.ListView()
        # mesh_available.set_items(["bottle", "can"])
        self._meshes_used = gui.ListView()
        # mesh_used.set_items(["can_0", "can_1", "can_1", "can_1"])
        add_mesh_button = gui.Button("Add Mesh")
        remove_mesh_button = gui.Button("Remove Mesh")
        add_mesh_button.set_on_clicked(self._add_mesh)
        remove_mesh_button.set_on_clicked(self._remove_mesh)
        annotation_objects.add_child(self._meshes_available)
        annotation_objects.add_child(add_mesh_button)
        annotation_objects.add_child(self._meshes_used)
        annotation_objects.add_child(remove_mesh_button)
        self._settings_panel.add_child(annotation_objects)

        self._scene_control = gui.CollapsableVert("Scene Control", 0.33 * em,
                                                  gui.Margins(em, 0, 0, 0))
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

        self._load_pre_annotation_button = gui.Button("Load Previous gt")
        self._load_pre_annotation_button.horizontal_padding_em = 0.8
        self._load_pre_annotation_button.vertical_padding_em = 0
        self._load_pre_annotation_button.set_on_clicked(self._on_load_previous_annotation)
        self._load_next_annotation_button = gui.Button("Load Next gt")
        self._load_next_annotation_button.horizontal_padding_em = 0.8
        self._load_next_annotation_button.vertical_padding_em = 0
        self._load_next_annotation_button.set_on_clicked(self._on_load_next_annotation)

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

        h = gui.Horiz(0.4 * em)  # row 1
        h.add_stretch()
        h.add_child(self._load_pre_annotation_button)
        h.add_child(self._load_next_annotation_button)
        h.add_stretch()
        self._scene_control.add_child(h)



        self._settings_panel.add_child(self._scene_control)
        # refine_position = gui.Button("Refine position")
        # refine_position.set_on_clicked(self._on_refine)
        generate_save_annotation = gui.Button("generate annotation - save/update")
        generate_save_annotation.set_on_clicked(self._on_generate)
        # self._scene_control.add_child(refine_position)
        self._scene_control.add_child(generate_save_annotation)

        fill_annotation = gui.Button("Fill previous empty annotation")
        fill_annotation.set_on_clicked(self._on_fill_annotation_from_last_avaiable_frame)
        self._scene_control.add_child(fill_annotation)

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

        # set callbacks for key control
        self.im_scene_widget.set_on_key(self._transform)
        self.pc_scene_widget.set_on_key(self._transform)

        self._left_shift_modifier = False

    def _update_img_numbers(self):
        self._image_number.text = "Image: " + f'{self._annotation_scene.image_num:06}'

    def _transform(self, event):
        if event.is_repeat:
            return gui.Widget.EventCallbackResult.HANDLED

        if event.key == gui.KeyName.LEFT_SHIFT:
            if event.type == gui.KeyEvent.DOWN:
                self._left_shift_modifier = True
            elif event.type == gui.KeyEvent.UP:
                self._left_shift_modifier = False
            return gui.Widget.EventCallbackResult.HANDLED

        # if ctrl is pressed then increase translation and angle values
        global dist, deg
        if event.key == gui.KeyName.LEFT_CONTROL:
            if event.type == gui.KeyEvent.DOWN:
                dist = 0.02
                deg = 5
            elif event.type == gui.KeyEvent.UP:
                dist = 0.05
                deg = 30
            return gui.Widget.EventCallbackResult.HANDLED

        if event.key == gui.KeyName.ALT:
            if event.type == gui.KeyEvent.DOWN:
                dist = 0.005
                deg = 1
            elif event.type == gui.KeyEvent.UP:
                dist = 0.05
                deg = 30
            return gui.Widget.EventCallbackResult.HANDLED

        # Change camera view
        if event.key >= 49 and event.key <= 56:
            if event.type == gui.KeyEvent.DOWN:
                view_id = event.key - 49
                if view_id < len(self.camera_names):
                    print(f'Change to camera_{view_id+1:02d}')
                    self.active_camera_view = view_id
                    self._update_rgb_views()
                    return gui.Widget.EventCallbackResult.HANDLED
                else:
                    return gui.Widget.EventCallbackResult.HANDLED
            else:
                return gui.Widget.EventCallbackResult.HANDLED

        # if no active_mesh selected print error
        if self._meshes_used.selected_index == -1:
            self._on_error("No objects are selected in scene meshes")
            return gui.Widget.EventCallbackResult.HANDLED

        def move(x, y, z, rx, ry, rz):
            self._annotation_changed = True

            objects = self._annotation_scene.get_objects()
            active_obj = objects[self._meshes_used.selected_index]

            T_ci_to_master = np.linalg.inv(self.extrinsics[self.camera_names[self.active_camera_view]])
            T_master_to_c0 = self.extrinsics[self.camera_names[0]]
            T_ci_to_c0 = T_ci_to_master @ T_master_to_c0

            # translation or rotation
            if x != 0 or y != 0 or z != 0:
                ci_h_transform = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
            else:  # elif rx!=0 or ry!=0 or rz!=0:
                c0_center = active_obj.obj_geometry.get_center()
                ci_center = T_ci_to_c0 @ np.append(c0_center, 1)
                ci_center = ci_center[:3]

                ci_T_neg = np.vstack((np.hstack((np.identity(3), -ci_center.reshape(3, 1))), [0, 0, 0, 1]))
                ci_T_pos = np.vstack((np.hstack((np.identity(3), ci_center.reshape(3, 1))), [0, 0, 0, 1]))

                rot_mat_obj_center = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
                R = np.vstack((np.hstack((rot_mat_obj_center, [[0], [0], [0]])), [0, 0, 0, 1]))
                ci_h_transform = np.matmul(ci_T_pos, np.matmul(R, ci_T_neg))

            h_transform = np.linalg.inv(T_ci_to_c0) @ ci_h_transform @ T_ci_to_c0

            active_obj.obj_geometry.transform(h_transform)
            # center = active_obj.obj_geometry.get_center()
            self.pc_scene_widget.scene.remove_geometry(active_obj.obj_name)
            self.pc_scene_widget.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                                    self.settings.annotation_obj_material,
                                                    add_downsampled_copy_for_fast_rendering=True)
            # update values stored of object
            active_obj.transform = np.matmul(h_transform, active_obj.transform)
            self._update_rgb_views()

        if event.type == gui.KeyEvent.DOWN:  # only move objects with down strokes
            # Refine
            if event.key == gui.KeyName.R:
                self._on_refine()
            # Translation
            if not self._left_shift_modifier:
                if event.key == gui.KeyName.J:
                    print("J pressed: translate to left")
                    move(-dist, 0, 0, 0, 0, 0)
                elif event.key == gui.KeyName.L:
                    print("L pressed: translate to right")
                    move(dist, 0, 0, 0, 0, 0)
                elif event.key == gui.KeyName.K:
                    print("K pressed: translate down")
                    move(0, dist, 0, 0, 0, 0)
                elif event.key == gui.KeyName.I:
                    print("I pressed: translate up")
                    move(0, -dist, 0, 0, 0, 0)
                elif event.key == gui.KeyName.U:
                    print("U pressed: translate furthur")
                    move(0, 0, dist, 0, 0, 0)
                elif event.key == gui.KeyName.O:
                    print("O pressed: translate closer")
                    move(0, 0, -dist, 0, 0, 0)
            # Rotation - keystrokes are not in same order as translation to make movement more human intuitive
            else:
                print("Left-Shift is clicked; rotation mode")
                if event.key == gui.KeyName.O:
                    print("O pressed: rotate CW")
                    move(0, 0, 0, 0, 0, deg * np.pi / 180)
                elif event.key == gui.KeyName.U:
                    print("U pressed: rotate CCW")
                    move(0, 0, 0, 0, 0, -deg * np.pi / 180)
                elif event.key == gui.KeyName.J:
                    print("J pressed: rotate towards left")
                    move(0, 0, 0, 0, deg * np.pi / 180, 0)
                elif event.key == gui.KeyName.L:
                    print("L pressed: rotate towards right")
                    move(0, 0, 0, 0, -deg * np.pi / 180, 0)
                elif event.key == gui.KeyName.K:
                    print("K pressed: rotate downwards")
                    move(0, 0, 0, deg * np.pi / 180, 0, 0)
                elif event.key == gui.KeyName.I:
                    print("I pressed: rotate upwards")
                    move(0, 0, 0, -deg * np.pi / 180, 0, 0)

        return gui.Widget.EventCallbackResult.HANDLED

    def _on_refine(self):
        self._annotation_changed = True

        # if no active_mesh selected print error
        if self._meshes_used.selected_index == -1:
            self._on_error("No objects are highlighted in scene meshes")
            return gui.Widget.EventCallbackResult.HANDLED

        target = self._annotation_scene.annotation_scene
        objects = self._annotation_scene.get_objects()
        active_obj = objects[self._meshes_used.selected_index]
        source = active_obj.obj_geometry

        trans_init = np.identity(4)
        threshold = 0.004
        radius = 0.002
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        reg = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                                          o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                          o3d.pipelines.registration.ICPConvergenceCriteria(
                                                              max_iteration=50))

        active_obj.obj_geometry.transform(reg.transformation)
        # active_obj.obj_geometry.paint_uniform_color([0,1,0])  # Debug
        self.pc_scene_widget.scene.remove_geometry(active_obj.obj_name)
        self.pc_scene_widget.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                                self.settings.annotation_obj_material,
                                                add_downsampled_copy_for_fast_rendering=True)
        active_obj.transform = np.matmul(reg.transformation, active_obj.transform)
        self._update_rgb_views()

    def _on_generate(self):
        for camera in self.camera_names:
            json_6d_path = os.path.join(self.scene_path, camera, "scene_gt.json")

            image_num = self._annotation_scene.image_num

            if os.path.exists(json_6d_path):
                with open(json_6d_path, "r") as gt_scene:
                    gt_6d_pose_data = json.load(gt_scene)
            else:
                gt_6d_pose_data = {}

            # wrtie/update "scene_gt.json"
            with open(json_6d_path, 'w+') as gt_scene:
                view_angle_data = list()
                for obj in self._annotation_scene.get_objects():
                    T_cam_to_master = np.linalg.inv(self.extrinsics[camera])
                    T_master_to_cam0 = self.extrinsics[self.camera_names[0]]
                    transform_cam_to_object = T_cam_to_master @ T_master_to_cam0 @ obj.transform
                    translation = list(transform_cam_to_object[0:3, 3] * 1000)  # convert meter to mm
                    obj_id = model_names.index(obj.obj_name[:-2]) + 1  # assuming max number of object of same object 10
                    obj_data = {
                        "cam_R_m2c": transform_cam_to_object[0:3, 0:3].tolist(),  # rotation matrix
                        "cam_t_m2c": translation,  # translation
                        "obj_id": obj_id
                    }
                    view_angle_data.append(obj_data)
                gt_6d_pose_data[str(image_num)] = view_angle_data
                json.dump(gt_6d_pose_data, gt_scene)

        self._annotation_changed = False
        print('annoation saved')

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

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

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

    def _add_mesh(self):
        meshes = self._annotation_scene.get_objects()
        meshes = [i.obj_name for i in meshes]

        if self._meshes_available.selected_index == -1:
            print('no obj selected')
            return

        object_geometry = o3d.io.read_point_cloud(ply_model_paths[self._meshes_available.selected_index + 1])
        object_geometry.points = o3d.utility.Vector3dVector(
            np.array(object_geometry.points) / 1000)  # convert mm to meter
        init_trans = np.identity(4)
        center = self._annotation_scene.annotation_scene.get_center()
        center[2] -= 0.2
        init_trans[0:3, 3] = center
        object_geometry.transform(init_trans)
        new_mesh_instance = self._obj_instance_count(self._meshes_available.selected_value, meshes)
        new_mesh_name = str(self._meshes_available.selected_value) + '_' + str(new_mesh_instance)
        self.pc_scene_widget.scene.add_geometry(new_mesh_name, object_geometry, self.settings.annotation_obj_material,
                                                add_downsampled_copy_for_fast_rendering=True)
        self._annotation_scene.add_obj(object_geometry, new_mesh_name, new_mesh_instance, transform=init_trans)
        meshes = self._annotation_scene.get_objects()  # update list after adding current object
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)
        self._meshes_used.selected_index = len(meshes) - 1
        self._update_rgb_views()

    def _remove_mesh(self):
        if not self._annotation_scene.get_objects():
            print("There are no object to be deleted.")
            return
        meshes = self._annotation_scene.get_objects()
        active_obj = meshes[self._meshes_used.selected_index]
        self.pc_scene_widget.scene.remove_geometry(active_obj.obj_name)  # remove mesh from scene
        self._annotation_scene.remove_obj(self._meshes_used.selected_index)  # remove mesh from class list
        # update list after adding removing object
        meshes = self._annotation_scene.get_objects()  # get new list after deletion
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)
        self._update_rgb_views()

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

        self._update_rgb_views()
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

    def _update_rgb_views(self):
        print('updating rgb views...')
        tt = time.time()
        obj_ext_poses = {}
        objs = self._annotation_scene.get_objects()
        for obj in objs:
            obj_id = model_names.index(obj.obj_name[:-2]) + 1
            pose = obj.transform
            pose = self.extrinsics[self.camera_names[0]] @ pose
            if obj_id in obj_ext_poses:
                obj_ext_poses[obj_id].append(pose)
            else:
                obj_ext_poses[obj_id] = [pose]

        imgs = []
        for i, camera_name in enumerate(self.camera_names):
            if len(objs) > 0:
                # change pose to corresponding camera view
                dict_id_poses = {}
                for obj_id, pose in obj_ext_poses.items():
                    dict_id_poses[obj_id] = np.linalg.inv(self.extrinsics[camera_name]) @ pose

                py_rendered_im = render_obj_pose(self.py_renderers[i], dict_id_poses)
                im = overlay_imgs(self.rgb_imgs[i], py_rendered_im, 1, 0.8)
            else:
                im = self.rgb_imgs[i]
            im = cv2.putText(im, camera_name[:9], (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
            if i == self.active_camera_view:
                im = add_border(im)
            imgs.append(im)
        img = collage_imgs(imgs, num_rows=3)
        img_geo = o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.im_scene_widget.scene.set_background([0, 0, 0, 0], img_geo)
        print('rgb view updated')

    def update_obj_list(self):
        self._meshes_available.set_items(model_names)

    def _check_changes(self):
        if self._annotation_changed:
            self._on_error(
                "Annotation changed but not saved. If you want to ignore the changes click the navigation button again.")
            self._annotation_changed = False
            return True
        else:
            return False

    def _on_next_image(self):
        if self._check_changes():
            return

        if self._annotation_scene.image_num + 1 >= len(
                next(os.walk(os.path.join(self.scene_path, f'{self.camera_names[0]}', 'depth')))[
                    2]):  # 2 for files which here are the how many depth images
            self._on_error("There is no next image.")
            return
        self.scene_load(self._annotation_scene.image_num + 1)

    def _on_previous_image(self):
        if self._check_changes():
            return

        if self._annotation_scene.image_num - 1 < 0:
            self._on_error("There is no image number before image 0.")
            return
        self.scene_load(self._annotation_scene.image_num - 1)

    def _on_load_previous_annotation(self):
        if self._annotation_scene.image_num - 1 < 0:
            self._on_error("There is no image number before image 0.")
            return
        self._load_annotation(self._annotation_scene.image_num - 1)
        self._update_rgb_views()
        self._annotation_changed = True
        self._meshes_used.selected_index = 0

    def _on_load_next_annotation(self):
        if self._annotation_scene.image_num + 1 >= len(
                next(os.walk(os.path.join(self.scene_path, f'{self.camera_names[0]}', 'depth')))[
                    2]):  # 2 for files which here are the how many depth images
            self._on_error("There is no next image.")
            return
        self._load_annotation(self._annotation_scene.image_num + 1)
        self._update_rgb_views()
        self._annotation_changed = True
        self._meshes_used.selected_index = 0

    def _on_fill_annotation_from_last_avaiable_frame(self):
        scene_gt_path = os.path.join(self.scene_path, f'{self.camera_names[0]}', 'scene_gt.json')
        if not os.path.exists(scene_gt_path):
            print('no scene_gt.json')
            return

        with open(scene_gt_path) as scene_gt_file:
            data = json.load(scene_gt_file)

        max_im_num = max([int(k) for k in data.keys() if int(k) < self._annotation_scene.image_num])
        print('copying annoation from frame', max_im_num, 'until frame', self._annotation_scene.image_num)

        for camera_name in self.camera_names:
            scene_gt_path = os.path.join(self.scene_path, camera_name, 'scene_gt.json')
            with open(scene_gt_path) as scene_gt_file:
                data = json.load(scene_gt_file)

            for i in range(max_im_num + 1, self._annotation_scene.image_num + 1):
                data[str(i)] = data[str(max_im_num)]

            with open(scene_gt_path, 'w') as scene_gt_file:
                json.dump(data, scene_gt_file)

        self._load_annotation(self._annotation_scene.image_num)
        self._update_rgb_views()


def main():
    parser = argparse.ArgumentParser(description='Annotation tool.')
    parser.add_argument('--scene_name', default='scene_2210232307_01')
    # parser.add_argument('--scene_name', default='scene_2210232348_wzq')
    parser.add_argument('--start_frame', type=int, default=366)

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
