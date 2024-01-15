import os, sys
import asyncio
import cv2
import time
import signal
import threading
import numpy as np
import open3d as o3d

from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Reader
from datetime import datetime
import uuid
import re
from PIL import Image
from scipy.spatial.transform import Rotation

import shutil, base64
from glob import glob
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import concurrent.futures
from multiprocessing import Manager, Queue, Process
from fastapi import Response
import subprocess

import nicegui
from nicegui import app, ui, Tailwind
from nicegui.events import ValueChangeEventArguments

from src.data_container import DataContainer
from src.data_loader import RosbagLoader, sort_list_of_dicts_by_key
from src.calibrator import Calibrator

#process_pool_executor = concurrent.futures.ProcessPoolExecutor()



class PersistentDialog:
    def __init__(self, msg:str, width:int=400, height:int=200, with_progress=False) -> None:
        with ui.dialog() as self.dialog, ui.card().style(f'width:{width}px; height:{height}px'):
            self.dialog.props('persistent')
            with ui.column().classes('absolute-center items-center w-full'):
                ui.spinner(size='3em')
                ui.label(msg)
                if with_progress:
                    self.progress_bar = ui.linear_progress(
                        value=0, size=10, show_value=False
                    ).style(f'width:{width-100}px').props('stripe')
                # self.subtask_label = ui.label('')
        self.dialog.open()

    def close(self):
        self.dialog.close()



class CalibrationGUI:
    def __init__(self) -> None:
        # Chessboard target configurations
        self.horizontal_inner_corners:int = 7
        self.vertical_inner_corners:int = 5
        self.chessboard_square_size:float = 0.1
        self.camera_model:str = 'pinhole'

        # Status parameters
        self.calibration_session_started = False
        self.calibrated = False

        # Sensors data
        self.sensors_data:DataContainer = None
        self.calibrator:Calibrator = None
        self.data_dir = 'sample_data/data_ladybug5'
        self.rosbag_target_time = 2.

        # Initialise temp directory
        try: shutil.rmtree('/tmp/calibrator_gui')
        except: pass
        self.temp_dir = f'/tmp/calibrator_gui/{uuid.uuid4()}'
        os.makedirs(self.temp_dir, exist_ok=True)

        # GUI parameters
        self.progress_value_queue = asyncio.Queue()
        self.current_task = 'None'
        self.progress = 0.
        self.single_rosbag_id:int = None
        self.current_rosbag_time:float = 0.
        self.rosbag_target_times:'list[float]' = []
        self.pcd_accumulation_frames = 0
        self.raw_images_already_undistorted = False
        self.reverse_2D_corners_switches:'dict[int, ui.switch]' = {}
        self.corners_3D_detector = 2
        self.sensor_fusion_attribute = 'depth'
        self.max_fusion_depth = 15.
        self.server_alive_dialog_shown = False

        # Calibration poses selection
        self.selected_poses_to_load:'dict[int, bool]' = {}
        self.selected_poses_to_load_checkboxes:'dict[int, ui.checkbox]' = {}
        self.selected_poses_to_load_buttons:'dict[int, ui.button]' = {}
        self.LiDAR_pose_checkboxes:'dict[int, ui.checkbox]' = {}
        self.selected_poses:'list[int]' = []
        self.selected_LiDAR:str = None
        self.auto_detect_interval_length:float = 1.
        self.auto_detected_poses_to_load:'dict[float, ui.checkbox]' = {}

        # Calibration stages
        class CalibrationStages:
            load_data:bool = False
            calibrate_intrinsics:bool = False
            detect_3d_corners:bool = False
            detect_2d_corners:bool = False
            calibrate_extrinsics:bool = False
        self.calibration_stages = CalibrationStages()

        # Initialise GUI
        ui.dark_mode(True)
        self.sidebar = ui.left_drawer().classes('bg-dark')
        self.make_sidebar()
        self.gui()



    def calibrate_LiDAR_camera(self):
        if len(self.calibrator.valid_LiDAR_poses) == 0:
            self.show_dialog('Please detect 3D corners first')
            return
        elif len(self.calibrator.valid_cam_poses) == 0:
            self.show_dialog('Please detect 2D corners first')
            return
        elif self.calibrator.cam_intrinsics is None:
            self.show_dialog('Please calibrate or load camera intrinsics first')
            return
        self.current_task = 'LiDAR-camera calibration'
        self.progress = 0.

        reversed_2D_corners_poses = []
        for k, v in self.reverse_2D_corners_switches.items():
            if v.value == True:
                reversed_2D_corners_poses.append(k)
        persistent_dialog = PersistentDialog(
            'Calibrating LiDAR-camera extrinsics',
            with_progress=True,
        )
        persistent_dialog.progress_bar.bind_value_from(self, 'progress')

        selected_poses = [i for i in range(self.calibrator.num_poses) if self.selected_poses_bool[i]]

        async def _worker():
            try:
                await self.calibrator.calibrate_LiDAR_camera_extrinsics(
                    selected_poses,
                    reversed_2D_corners_poses,
                    queue=self.progress_value_queue
                )
            except Exception as e:
                ui.notify('LiDAR-camera calibration failed', type='negative')
                persistent_dialog.close()
                raise e
                return
            # Plot re-projection errors
            for i, err in enumerate(self.calibrator.PnP_reprojection_errors):
                self.reprj_err_chart.options['series'][0]['data'][i]['y'] = 0.
                if err is not None:
                    self.reprj_err_chart.options['series'][0]['data'][i]['y'] = err
            # Plot mean re-projection error
            mean_reprj_err = np.average([err for err in self.calibrator.PnP_reprojection_errors if err is not None])
            mean_reprj_err_line = {
                'name': f'Average re-projection error = {mean_reprj_err:.3f}',
                'type': 'line',
                'color': '#009900',
                'data': [{
                    'x': i,
                    'y': mean_reprj_err,
                } for i in range(self.calibrator.num_poses)],
                'dashStyle': 'shortdot',
                'lineWidth': 3,
                'marker': {'enabled': False}
            }
            try:
                self.reprj_err_chart.options['series'][1] = mean_reprj_err_line
            except IndexError:
                self.reprj_err_chart.options['series'].append(mean_reprj_err_line)
            self.reprj_err_chart.update()

            # Show fusion and re-projection results
            self.LiDAR_to_cam_fusion_display_area.clear()
            with self.LiDAR_to_cam_fusion_display_area:
                with ui.grid(columns=1).classes('w-full'):
                    selected_pose = self.calibrator.valid_cam_poses[0]
                    coloured_pcd = self.calibrator.colour_point_cloud(
                        self.sensors_data.point_clouds[self.selected_LiDAR][selected_pose],
                        self.calibrator.undistorted_images[selected_pose],
                        self.calibrator.opt_cam_pose
                    )
                    self.show_point_cloud(coloured_pcd, height=800, colour_attribute='RGB', point_size=0.1)
                ui.html("<p>&nbsp<p>")
                self.LiDAR_to_cam_fusion_images_area = ui.grid(columns=2)
                with self.LiDAR_to_cam_fusion_images_area:
                    for i in range(self.calibrator.num_poses):
                        reproj_img_dir = self.calibrator.PnP_reprojection_images[i]
                        fusion_img_dir = self.calibrator.fusion_images[i]
                        if reproj_img_dir is None or fusion_img_dir is None: continue
                        with ui.image(fusion_img_dir).classes('w-full').props('fit=contain'):
                            ui.label(f'Pose {i}').classes('absolute-bottom text-subtitle2 text-center')
                        ui.image(reproj_img_dir).classes('w-full').props('fit=contain')
            persistent_dialog.close()
            ui.notify('LiDAR-camera calibration completed', type='positive')
            self.calibration_stages.calibrate_extrinsics = True
            self.calibrated = True

        ui.timer(0.01, _worker, once=True)
    


    def update_LiDAR_to_cam_fusion_images(self):
        if not self.calibrated: return
        self.persistent_dialog = PersistentDialog(
            'Updating fusion images',
            with_progress=True,
        )
        self.persistent_dialog.progress_bar.bind_value_from(self, 'progress')

        async def _worker():
            self.LiDAR_to_cam_fusion_images_area.clear()
            with self.LiDAR_to_cam_fusion_images_area:
                for i in self.calibrator.valid_cam_poses:
                    reproj_img_dir = self.calibrator.PnP_reprojection_images[i]
                    if reproj_img_dir is None: continue
                    fusion_img_dir = self.calibrator.project_point_cloud_onto_image(
                        self.sensors_data.point_clouds[self.selected_LiDAR][i],
                        self.sensors_data.images[i],
                        self.calibrator.opt_cam_pose,
                        depth_max=self.max_fusion_depth,
                        fusion_attribute=self.sensor_fusion_attribute,
                    )
                    with ui.image(fusion_img_dir).classes('w-full').props('fit=contain'):
                        ui.label(f'Pose {i}').classes('absolute-bottom text-subtitle2 text-center')
                    ui.image(reproj_img_dir).classes('w-full').props('fit=contain')
                    await self.progress_value_queue.put((i+1)/len(self.calibrator.valid_cam_poses))
            self.persistent_dialog.close()

        ui.timer(0.001, _worker, once=True)




    def save_calibration(self):
        json_dir = os.path.join(self.data_dir, 'calib', self.sensors_data.cam_name, self.selected_LiDAR, 'extrinsics.json')
        pass





    def show_all_chessboard_poses(self):
        if len(self.selected_poses) > 0:
            selected_poses = self.selected_poses
        else:
            selected_poses = self.calibrator.valid_LiDAR_poses

        pcd = self.sensors_data.point_clouds[self.selected_LiDAR][selected_poses[0]]
        I = pcd[:, 3]
        I /= max(I)
        points = pcd[:, :3]
        colors = plt.get_cmap('gray')(I)[:,:3]

        with ui.scene(height=600, grid=False).classes('w-full') as scene_3D:
            # Show all the 3D chessboard poses
            for i, pose_id in enumerate(selected_poses):
                chessboard_pose = self.calibrator.all_chessboard_poses[pose_id]
                # Chessboard bounding box
                chessboard = scene_3D.box(
                    width=(self.horizontal_inner_corners + 1) * self.chessboard_square_size,
                    height=(self.vertical_inner_corners + 1) * self.chessboard_square_size,
                    depth=0.04,
                    wireframe=True,
                )
                colour_decimal = plt.get_cmap('tab20')(i)[:3]
                colour = [int(c * 255) for c in colour_decimal]
                chessboard.color = f'#{colour[0]:02x}{colour[1]:02x}{colour[2]:02x}'
                chessboard.move(x=chessboard_pose[0, 3], y=chessboard_pose[1, 3], z=chessboard_pose[2, 3])
                chessboard.rotate_R(chessboard_pose[:3, :3].tolist())
                # Pose id indicator
                pose_id_label = scene_3D.text3d(
                    f'Pose {pose_id}',
                    f'background: rgba({colour[0]}, {colour[1]}, {colour[2]}, 0.2); color: rgba(0, 0, 0, 1); border-radius: 5px; padding: 5px'
                )
                pose_id_label.rotate_R(chessboard_pose[:3, :3].tolist())
                pose_id_label.move(x=chessboard_pose[0, 3], y=chessboard_pose[1, 3], z=chessboard_pose[2, 3])
                pose_id_label.scale(0.005)

            # Adjust camera angle
            chessboard_pose = self.calibrator.all_chessboard_poses[selected_poses[0]]
            scene_3D.move_camera(
                x = 0., y = 0., z = 0.2,
                look_at_x = chessboard_pose[0],
                look_at_y = chessboard_pose[1],
                look_at_z = chessboard_pose[2],
            )

            if self.calibrator.opt_cam_pose is not None:
                cam_pose = self.calibrator.opt_cam_pose
                cam = scene_3D.stl('./src/assets/camera.stl')
                cam.move(x=cam_pose[0, 3], y=cam_pose[1, 3], z=cam_pose[2, 3])
                cam.rotate_R(cam_pose[:3, :3].tolist())

            scene_3D.point_cloud(points=points.tolist(), colors=colors.tolist(), point_size=0.02)





    def make_LiDAR_camera_calibration_tab(self):
        self.tab_panels.set_value('LiDAR_cam_calib')
        try:
            if self.LiDAR_camera_calibration_tab_created: return
        except AttributeError: pass
        self.LiDAR_camera_calibration_tab_created = True

        self.selected_poses_bool = [True if i in self.calibrator.valid_cam_poses else False for i in range(self.calibrator.num_poses)]
        def handle_point_click(event):
            pose_id = event.point_x
            self.selected_poses_bool[pose_id] = not self.selected_poses_bool[pose_id]
            if self.selected_poses_bool[pose_id]:
                self.reprj_err_chart.options['series'][0]['data'][pose_id]['color'] = '#3874cb'
            else:
                self.reprj_err_chart.options['series'][0]['data'][pose_id]['color'] = '#a8a8a8'
            self.reprj_err_chart.update()

        with self.LiDAR_cam_calib_tab:
            ui.label('Selected calibration poses:').style('font-weight: bold')
            poses_disp_area = ui.grid(columns=1).classes('w-full')

            ui.html("<p>&nbsp<p>")
            with ui.grid(columns=1):
                self.reprj_err_chart = ui.chart({
                    'title': {'text': 'PnP re-projection errors'},
                    'xAxis': [{'title': {'text': 'Pose'}}],
                    'yAxis': [{'title': {'text': 'Re-projection error [pixels]'}}],
                    'chart': {'type': 'column'},
                    'series': [{
                        'name': 'Re-projection error',
                        'type': 'column',
                        'colorByPoint': True,
                        'showInLegend': False,
                        'data': [{
                            'x':i,
                            'y':0.,
                            'color': '#3874cb' if self.selected_poses_bool[i] else '#a8a8a8'
                        } for i in range(self.calibrator.num_poses)]
                    }]
                }, on_point_click=lambda e: handle_point_click(e))
                ui.label('Outliers can be deselected by clicking on the corresponding bar in the chart above. Press re-calibrate button after removing unwanted outliers.')

            ui.html("<p>&nbsp<p>")
            self.LiDAR_to_cam_fusion_display_area = ui.grid(columns=1).classes('w-full')
            with poses_disp_area:
                self.show_all_chessboard_poses()




    def detect_2d_corners(self):
        self.current_task = '2D corners detection'
        self.progress = 0.

        persistent_dialog = PersistentDialog(
            'Detecting 2D corners from given images',
            with_progress=True,
        )
        persistent_dialog.progress_bar.bind_value_from(self, 'progress')
        if len(self.LiDAR_pose_checkboxes.values()) > 0:
            self.selected_poses = [i for i, v in self.LiDAR_pose_checkboxes.items() if v.value == True]
        if len(self.selected_poses) == 0:
            self.selected_poses = self.calibrator.valid_LiDAR_poses
        if len(self.selected_poses) == 0:
            self.selected_poses = [i for i in range(self.calibrator.num_poses)]

        try:
            self.corners_2d_detection_display_area
        except AttributeError:
            self.make_2d_corners_detection_tab()

        async def _worker():
            await self.calibrator.detect_2d_corners(
                selected_poses=self.selected_poses,
                queue=self.progress_value_queue
            )
            persistent_dialog.close()
            self.corners_2d_detection_display_area.clear()
            self.reverse_2D_corners_switches:'dict[int, ui.switch]' = {}
            with self.corners_2d_detection_display_area:
                for i in self.calibrator.valid_cam_poses:
                    img = self.calibrator.corners_2d_demo_dir[i]
                    with ui.column():
                        self.reverse_2D_corners_switches[i] = ui.switch(
                            text=f'Reverse corners order of pose {i}',
                            value=False,
                        )
                        with ui.image(img).classes('w-full').props('fit=contain'):
                            ui.label(f'Pose {i}').classes('absolute-bottom text-subtitle2 text-center')
            self.calibration_stages.detect_2d_corners = True

        ui.timer(0.01, _worker, once=True)



    def make_2d_corners_detection_tab(self):
        self.stepper.set_value('Detect 2D corners')
        self.tab_panels.set_value('2D_chessboard')
        try:
            self.corners_2d_detection_display_area
        except AttributeError:
            self.detect_2d_corners_tab.clear()
            with self.detect_2d_corners_tab:
                self.corners_2d_detection_display_area = ui.grid(columns=2).classes('w-full')
                self.bio_retina_switch.bind_value_to(self.calibrator, 'use_bio_retina')



    def detect_3d_corners(self):
        if not self.calibration_session_started:
            return
        LiDAR_name = self.selected_LiDAR
        if LiDAR_name == 'pandar40':
            self.roi_direction_slider.set_value(180.)
        self.current_task = '3D corners detection'
        self.progress = 0.

        persistent_dialog = PersistentDialog(
            'Detecting 3D corners from given point clouds',
            with_progress=True,
        )
        persistent_dialog.progress_bar.bind_value_from(self, 'progress')

        try:
            self.corners_3d_detection_display_area
        except AttributeError:
            self.make_3d_corners_detection_tab()

        async def _worker():
            await self.calibrator.detect_3d_corners(
                LiDAR_name,
                selected_poses=self.selected_poses,
                corners_3D_detector=self.corners_3D_detector,
                queue=self.progress_value_queue
            )
            persistent_dialog.close()

            self.corners_3d_detection_display_area.clear()
            try:
                pcd_id = self.calibrator.valid_LiDAR_poses[0]
            except:
                pcd_id = 0
            with self.corners_3d_detection_display_area:
                self.show_point_cloud(
                    self.sensors_data.point_clouds[LiDAR_name][pcd_id],
                    chessboard_poses = self.calibrator.all_chessboard_poses,
                    all_chessboard_3d_corners=self.calibrator.all_chessboard_3d_corners,
                )
                #self.show_all_chessboard_poses()
                self.LiDAR_pose_checkboxes:'dict[ui.checkbox]' = {
                    pose_id: None for pose_id in self.calibrator.valid_LiDAR_poses
                }
                def display_slot(i, img):
                    btn = ui.button().props('flat padding="none"').classes('w-full')
                    cost = self.calibrator.corners_3D_costs[i]
                    with btn:
                        self.LiDAR_pose_checkboxes[i] = ui.checkbox(
                            f'Pose {i}, cost = {cost:.2f}', value=True if cost < 200. else False
                        ).classes('w-full')
                        with ui.image(img).classes('w-full').props('fit=contain'):
                            ui.label('Excluded').classes(
                                'absolute-full text-subtitle2 flex flex-center'
                            ).bind_visibility_from(self, 'LiDAR_pose_checkboxes', lambda checkboxes: checkboxes[i].value==False)
                    btn.on(
                        'click',
                        lambda: self.LiDAR_pose_checkboxes[i].set_value(not self.LiDAR_pose_checkboxes[i].value)
                    )
                with ui.grid(columns=4).classes('w-full'):
                    for i, img in enumerate(self.calibrator.opt_result_vis):
                        if img is None: continue
                        with ui.column(): display_slot(i, img)
                            

            self.calibration_stages.detect_3d_corners = True

        ui.timer(0.01, _worker, once=True)



    def make_3d_corners_detection_tab(self):
        self.tab_panels.set_value('3D_chessboard')
        try:
            if self.created_3d_corners_detection_tab: return
        except AttributeError: pass
        self.created_3d_corners_detection_tab = True
        self.detect_3d_corners_tab.clear()

        def update_demo_point_cloud():
            if inspection_expandable.value == False: return
            LiDAR_name:str = demo_LiDAR_name_selector.value,
            LiDAR_name = LiDAR_name[0]
            if LiDAR_name == 'pandar40':
                self.roi_direction_slider.set_value(180.)
            pose_id:int = self.LiDAR_pose_id_selector.value
            demo_pcd_input_display_area.clear()
            with demo_pcd_input_display_area:
                self.show_point_cloud(self.sensors_data.point_clouds[LiDAR_name][pose_id], cmap='winter')
            demo_pcd_input_display_area.update()

        def demo_3d_corners_detection():
            persistent_dialog = PersistentDialog('Detecting 3D corners from given point cloud')
            LiDAR_name:str = demo_LiDAR_name_selector.value,
            LiDAR_name = LiDAR_name[0]
            pose_id:int = self.LiDAR_pose_id_selector.value

            async def _worker():
                await self.calibrator.detect_3d_corners(
                    LiDAR_name, selected_poses=[pose_id],
                    corners_3D_detector=self.corners_3D_detector,
                    demo=True
                )
                chessboard_detection_display_area.clear()
                with chessboard_detection_display_area:
                    self.show_point_cloud(
                        self.sensors_data.point_clouds[LiDAR_name][pose_id],
                        chessboard_poses = [self.calibrator.all_chessboard_poses[pose_id]],
                        all_chessboard_3d_corners = [self.calibrator.all_chessboard_3d_corners[pose_id]],
                        cmap = 'winter'
                    )
                demo_LiDAR_image_detection_display_area.clear()
                with demo_LiDAR_image_detection_display_area:
                    with ui.grid(columns=3).classes('w-full'):
                        with ui.column():
                            ui.label('Corners detection from LiDAR image')
                            ui.image(self.calibrator.showcase['LiDAR_image']).classes('w-full').props('fit=contain')
                        with ui.column():
                            ui.label('Before optimisation')
                            ui.image(self.calibrator.showcase['before_opt']).classes('w-full').props('fit=contain')
                        with ui.column():
                            ui.label('After optimisation')
                            ui.image(self.calibrator.showcase['after_opt']).classes('w-full').props('fit=contain')
                inspection_expandable.set_value(True)
                update_demo_point_cloud()
                persistent_dialog.close()

            ui.timer(0.01, _worker, once=True)

        with self.detect_3d_corners_tab:
            #ui.label('3D corners detection inspection area').style('font-weight: bold')
            with ui.grid(columns=2).classes('w-full'):
                with ui.grid(columns=2).classes('w-full'):
                    demo_LiDAR_name_selector = ui.select(
                        label='LiDAR name',
                        options=self.sensors_data.lidar_names,
                        value = 'os128' if 'os128' in self.sensors_data.lidar_names else self.sensors_data.lidar_names[0],
                        on_change=update_demo_point_cloud
                    ).classes('w-full')

                    self.LiDAR_pose_id_selector = ui.select(
                        label='Pose ID',
                        options=[i for i in range(len(self.sensors_data.images))],
                        value=0,
                        on_change=update_demo_point_cloud
                    ).classes('w-full')

                ui.button(
                    'Inspect 3D corners detection',
                    on_click=lambda:demo_3d_corners_detection()
                ).classes('w-full')

            ui.html("<p>&nbsp<p>")
            inspection_expandable = ui.expansion('3D corners detection inspection area').classes('w-full')
            with inspection_expandable:
                with ui.grid(columns=2).classes('w-full'):
                    demo_pcd_input_display_area = ui.grid().classes('w-full').style('height:600px')
                    chessboard_detection_display_area = ui.grid().classes('w-full').style('height:600px')
                ui.html("<p>&nbsp<p>")
                demo_LiDAR_image_detection_display_area = ui.grid().classes('w-full')

            ui.html("<p>&nbsp<p>"); ui.separator(); ui.html("<p>&nbsp<p>")
            ui.label('3D corners detection from LiDAR point cloud').style('font-weight: bold')

            self.LiDAR_name_selector.set_options(
                self.sensors_data.lidar_names,
                value = 'os128' if 'os128' in self.sensors_data.lidar_names else self.sensors_data.lidar_names[0],
            )
            self.LiDAR_name_selector.update()

            ui.html("<p>&nbsp<p>")
            self.corners_3d_detection_display_area = ui.grid(columns=1).classes('w-full')

        def inspection_expandable_handler(event):
            if event.args == True:
                update_demo_point_cloud()
        inspection_expandable.on('update:model-value', lambda event: inspection_expandable_handler(event))
    



    def export_pcd_chessboard_labels(self):
        if len(self.LiDAR_pose_checkboxes.keys()) == 0:
            self.show_dialog('Please detect 3D chessboard corners first')
            return
        else:
            self.selected_poses = [i for i, v in self.LiDAR_pose_checkboxes.items() if v.value == True]
        if len(self.selected_poses) == 0:
            self.show_dialog('Please select at least one pose')
            return
        self.show_persistent_dialog('Exporting point clouds with chessboard labels')
        self.calibrator.export_pcd_chessboard_labels(self.selected_poses)
        ui.notify('Point clouds with chessboard labels exported', type='positive')
        self.persistent_dialog.close()




    async def load_sensors_data(self):
        try:
            selected_camera = await self.image_stats_grid.get_selected_row()
            selected_lidars = await self.pcd_stats_grid.get_selected_rows()
        except AttributeError:
            self.show_dialog('Please load some data first')
            return

        self.selected_poses = []
        for pose_id, cb in self.selected_poses_to_load_checkboxes.items():
            if cb.value == True:
                self.selected_poses.append(pose_id)

        init_check_err_msg = ''
        if selected_camera == None:
            init_check_err_msg = 'Please select one image topic'
        elif len(selected_lidars) == 0:
            init_check_err_msg = 'Please select one or more point cloud topics'
        if init_check_err_msg != '':
            self.show_dialog(init_check_err_msg)
            return

        self.calibration_session_started = True
        self.sensors_data = DataContainer(selected_camera, selected_lidars, self.data_dir)
        self.calibrator = Calibrator(
            self.sensors_data,
            self.horizontal_inner_corners,
            self.vertical_inner_corners,
            self.chessboard_square_size,
            self.temp_dir,
            self.raw_images_already_undistorted
        )
        self.roi_direction_slider.bind_value_to(self.calibrator, 'roi_direction')
        self.roi_width_slider.bind_value_to(self.calibrator, 'roi_angular_width')

        self.edit_data_dir_button.disable()
        self.chessboard_configs_expansion.set_value(False)
        self.calibration_stages.load_data = True
        self.stepper.next()




    # Camera intrinsics calibration

    def load_camera_intrinsics(self):
        self.make_calibrate_intrinsics_tab()
        json_dir = os.path.join(self.data_dir, 'calib', self.sensors_data.cam_name, 'intrinsics.json')
        yaml_dir = os.path.join(self.data_dir, 'calib', self.sensors_data.cam_name, 'intrinsics.yaml')

        if not os.path.exists(json_dir) and not os.path.exists(yaml_dir):
            self.show_dialog('No camera intrinsics json or yaml file found on the selected data directory')
            return
        if self.calibrator is None:
            self.calibrator = Calibrator(
                self.sensors_data,
                self.horizontal_inner_corners,
                self.vertical_inner_corners,
                self.chessboard_square_size,
                self.temp_dir,
                self.raw_images_already_undistorted
            )
            self.roi_direction_slider.bind_value_to(self.calibrator, 'roi_direction')
            self.roi_width_slider.bind_value_to(self.calibrator, 'roi_angular_width')

        if os.path.exists(json_dir):
            try:
                self.calibrator.cam_intrinsics.load_json(json_dir)
            except:
                self.show_dialog('Camera intrinsics json file is corrupted or not of the correct format')
                return
        elif os.path.exists(yaml_dir):
            try:
                self.calibrator.cam_intrinsics.load_yaml(yaml_dir)
            except:
                self.show_dialog('Camera intrinsics yaml file is corrupted or not of the correct format')
                return

        persistent_dialog = PersistentDialog('Loading camera intrinsics')

        async def _worker():
            self.cam_intrinsic_display_area.clear()
            intrinsics_display_data = {
                'Focal length': {
                    'fx': self.calibrator.cam_intrinsics.FocalLength[0],
                    'fy': self.calibrator.cam_intrinsics.FocalLength[1]
                },
                'Principle point': {
                    'cx': self.calibrator.cam_intrinsics.PrinciplePoint[0],
                    'cy': self.calibrator.cam_intrinsics.PrinciplePoint[1]
                },
                'Skew': {
                    's': self.calibrator.cam_intrinsics.Skew,
                },
                'Radial distortion': {
                    'k1': self.calibrator.cam_intrinsics.RadialDistortion[0],
                    'k2': self.calibrator.cam_intrinsics.RadialDistortion[1],
                    'k3': self.calibrator.cam_intrinsics.RadialDistortion[2]
                },
                'Tangential distortion': {
                    'p1': self.calibrator.cam_intrinsics.TangentialDistortion[0],
                    'p2': self.calibrator.cam_intrinsics.TangentialDistortion[1]
                },
                'Image size': {
                    'width': self.calibrator.cam_intrinsics.ImageSize[0],
                    'height': self.calibrator.cam_intrinsics.ImageSize[1]
                },
            }
            tree_data = []
            for key, val in intrinsics_display_data.items():
                if isinstance(val, dict):
                    tree_data.append({
                        'id': key,
                        'children': [{'id': k, 'content': f'{v}'} for k, v in val.items()]
                    })
                else:
                    tree_data.append({
                        'id': key, 'content': f'{val}'
                    })

            def demo_undistort_image(img_id:int):
                self.calibrator.cam_intrinsics.undistort_image(self.sensors_data.images[img_id], demo=True)
                imgs_display_area.clear()
                with imgs_display_area:
                    with ui.image(self.calibrator.cam_intrinsics.demo_undistort['raw']):
                        ui.label('Raw image').classes('absolute-bottom text-subtitle2 text-center')
                    with ui.image(self.calibrator.cam_intrinsics.demo_undistort['undistorted']):
                        ui.label('Undistorted image').classes('absolute-bottom text-subtitle2 text-center')

            self.cam_intrinsic_display_area.clear()
            with self.cam_intrinsic_display_area:
                with ui.splitter(value=30) as splitter:
                    # with splitter.separator:
                    #     ui.icon('swap_horizontal_circle', size='2em', color='primary')
                    with splitter.before:
                        tree = ui.tree(tree_data, label_key='id').props('no-connectors').expand()
                    with splitter.after:
                        undst_img_selector = ui.select(
                            label='Image pose to undistort',
                            options=[i for i in range(len(self.sensors_data.images))],
                            value=0,
                            on_change=lambda: demo_undistort_image(undst_img_selector.value)
                        ).props('filled')
                        ui.html("<p>&nbsp<p>")
                        imgs_display_area = ui.grid(columns=2)#.classes('w-full')

            tree.add_slot('default-header', '''
                <span :props="props"><strong>{{ props.node.id }}</strong>: {{ props.node.content }}</span>
            ''')
            demo_undistort_image(undst_img_selector.value)
            self.calibration_stages.calibrate_intrinsics = True
            persistent_dialog.close()
            await asyncio.sleep(0.001)

        ui.timer(0.001, _worker, once=True)



    def make_calibrate_intrinsics_tab(self):
        self.tab_panels.set_value('cam_calib')
        try:
            if self.calibrate_intrinsic_tab_created: return
        except AttributeError: pass

        self.calibrate_intrinsic_tab_created = True
        self.calibrate_intrinsic_tab.clear()
        with self.calibrate_intrinsic_tab:
            ui.label('Camera intrinsics parameters').style('font-weight: bold')
            self.cam_intrinsic_display_area = ui.grid(columns=1).classes('w-full')




    # Load data from file

    def make_load_data_from_file_tab(self):
        # Read images
        try:
            cam_names = os.listdir(os.path.join(self.data_dir, 'calib'))
            if len(cam_names) == 0:
                self.show_dialog('No data found on the current data directory')
                return
        except FileNotFoundError:
            self.show_dialog('Data directory does not exist')
            return
        imgs_check_valid = False
        for cam_name in cam_names:
            img_paths = sorted(glob(os.path.join(self.data_dir, 'calib', cam_name, 'imgs/*.png')))
            if len(img_paths) > 0:
                imgs_check_valid = True
                break
        if not imgs_check_valid:
            self.show_dialog('No data found on the current data directory')
            return

        self.data_dir_input.disable()
        self.load_file_button.disable()
        self.load_rosbag_button.disable()
        self.edit_data_dir_button.enable()

        self.load_file_tab.clear()
        self.tab_panels.set_value('load_data')

        image_stats = {
            'columns': [
                {'headerName':'Camera name', 'field':'sensor-name', 'width':170, 'checkboxSelection':True},
                {'headerName':'N. images', 'field':'data-count', 'flex':1},
                {'headerName':'Mode', 'field':'colour', 'flex':1}, # Either 'RGB' or 'Gray'
                {'headerName':'Width', 'field':'width', 'flex':1},
                {'headerName':'Weight', 'field':'height', 'flex':1},
            ],
            'rows': []
        }
        pcd_stats = {
            'columns': [
                {'headerName':'LiDAR name', 'field':'sensor-name', 'width':170, 'checkboxSelection':True, 'headerCheckboxSelection':True},
                {'headerName':'N. point clouds', 'field':'data-count', 'flex':1},
                # {'headerName':'x-Rot correction [deg]', 'field':'x-rot', 'flex':1, 'editable':True},
                # {'headerName':'y-Rot correction [deg]', 'field':'y-rot', 'flex':1, 'editable':True},
                # {'headerName':'z-Rot correction [deg]', 'field':'z-rot', 'flex':1, 'editable':True},
            ],
            'rows': []
        }

        for cam_name in cam_names:
            img_paths = sorted(glob(os.path.join(self.data_dir, 'calib', cam_name, 'imgs/*.png')))
            sample_img = np.array(Image.open(img_paths[0]))
            #print(sample_img.shape); break
            if len(sample_img.shape) == 3 and sample_img.shape[2] == 3:
                colour = 'RGB'
            else:
                colour = 'Gray'
            image_stats['rows'].append({
                'sensor-name': cam_name,
                'data-count': len(img_paths),
                'width': sample_img.shape[1],
                'height': sample_img.shape[0],
                'colour': colour
            })
        image_stats['rows'] = sort_list_of_dicts_by_key(image_stats['rows'], 'sensor-name')

        selected_cam_name = cam_names[0]

        # Read and display images on camera selection
        async def load_images(cam_name):
            # selected_cam = await self.image_stats_grid.get_selected_row()
            # if selected_cam != None:
            #     cam_name = selected_cam['sensor-name']
            # else:
            #     return
            orig_img_paths = sorted(glob(os.path.join(self.data_dir, 'calib', cam_name, 'imgs/*.png')))
            temp_img_paths = []
            img_nums = []
            for orig_img_path in orig_img_paths:
                img_num = re.search('[0-9][0-9]\.png', orig_img_path)
                if img_num is not None:
                    img_num = int(img_num.group()[0:2])
                else:
                    img_num = re.search('[0-9]\.png', orig_img_path)
                    if img_num is None:
                        raise ValueError('Please check the naming of image files in the data directory')
                    else:
                        img_num = int(img_num.group()[0])

                temp_img_paths.append(os.path.join(self.temp_dir, f'{uuid.uuid1()}.png'))
                img_nums.append(img_num)
                shutil.copyfile(orig_img_path, temp_img_paths[-1])

            sorted_ids = np.argsort(img_nums)
            img_nums = [img_nums[i] for i in sorted_ids]
            temp_img_paths = [temp_img_paths[i] for i in sorted_ids]

            calib_data_display_area.clear()
            self.selected_poses_to_load = {}
            self.selected_poses_to_load_checkboxes = {}
            self.selected_poses_to_load_buttons = {}

            def display_slot(img_num, temp_img_path):
                btn = ui.button().props('flat padding="none"').classes('w-full')
                with btn:
                    with ui.column().classes('w-full'):
                        self.selected_poses_to_load_checkboxes[img_num] = ui.checkbox(
                            f'Pose {img_num}', value=True
                        ).classes('w-full')
                        with ui.image(temp_img_path).classes('w-full').props('fit=contain'):
                            ui.label('Excluded').classes(
                                'absolute-full text-subtitle2 flex flex-center'
                            ).bind_visibility_from(
                                self, 'selected_poses_to_load_checkboxes',
                                lambda checkboxes: checkboxes[img_num].value==False
                            )
                btn.on(
                    'click', 
                    lambda: self.selected_poses_to_load_checkboxes[img_num].set_value(
                        not self.selected_poses_to_load_checkboxes[img_num].value
                    )
                )

            with calib_data_display_area:
                for img_num, temp_img_path in zip(img_nums, temp_img_paths):
                    display_slot(img_num, temp_img_path)

        # Read point cloud .npy files
        async def load_pcds():
            # ui.timer(0.01, lambda: load_images(), once=True)
            selected_cam = await self.image_stats_grid.get_selected_row()
            if selected_cam != None:
                cam_name = selected_cam['sensor-name']
            else:
                return
            await load_images(cam_name)
            pcd_names = os.listdir(os.path.join(self.data_dir, 'calib', cam_name))
            pcd_names.remove('imgs')
            pcd_stats['rows'] = []
            for pcd_name in pcd_names:
                pcd_path_prefix = os.path.join(self.data_dir, 'calib', cam_name, pcd_name)
                if not os.path.isdir(pcd_path_prefix): continue
                pcd_paths = sorted(glob(os.path.join(pcd_path_prefix, '*.npy')))
                pcd_stats['rows'].append({
                    'sensor-name': pcd_name,
                    'data-count': len(pcd_paths),
                    'x-rot': 0,
                    'y-rot': 0,
                    'z-rot': 0,
                })
            pcd_stats['rows'] = sort_list_of_dicts_by_key(pcd_stats['rows'], 'sensor-name')
            self.pcd_stats_grid.options['rowData'] = pcd_stats['rows']
            self.pcd_stats_grid.update()
            self.pcd_stats_grid.call_api_method('selectAll')

        def toggle_all_poses_to_load(value):
            for cb in self.selected_poses_to_load_checkboxes.values():
                cb.set_value(value)

        with self.load_file_tab:
            with ui.grid(columns=2).classes('w-full'):
                self.image_stats_grid = ui.aggrid({
                    'defaultColDef': {'resizable':True, 'sortable':True},
                    'columnDefs': image_stats['columns'],
                    'rowData': image_stats['rows'],
                    'rowSelection': 'single',
                    'rowMultiSelectWithClick': True,
                    'animateRows': True
                }, theme='alpine-dark').classes('w-full').style(r'height: 350px')

                self.pcd_stats_grid = ui.aggrid({
                    'defaultColDef': {'resizable':True, 'sortable':True},
                    'columnDefs': pcd_stats['columns'],
                    'rowData': pcd_stats['rows'],
                    'rowSelection': 'multiple',
                    'rowMultiSelectWithClick': True,
                    'animateRows': True
                }, theme='alpine-dark').classes('w-full').style(r'height: 350px')

                ui.label('Select one camera sensor')
                ui.label('Select one or more LiDAR sensors')

            ui.html("<p>&nbsp<p>")
            ui.separator()
            ui.html("<p>&nbsp<p>")

            ui.label('Loaded images:').style('font-weight: bold')
            with ui.row():
                ui.button(
                    'Select all',
                    on_click=lambda: toggle_all_poses_to_load(True)
                ).bind_enabled_from(self, 'selected_poses_to_load_checkboxes', lambda v: len(v.values()) > 0)
                ui.button(
                    'Deselect all',
                    on_click=lambda: toggle_all_poses_to_load(False)
                ).bind_enabled_from(self, 'selected_poses_to_load_checkboxes', lambda v: len(v.values()) > 0)
            ui.html("<p>&nbsp<p>")
            calib_data_display_area = ui.grid(columns=4).classes('w-full')

        self.image_stats_grid.on('selectionChanged', lambda: load_pcds())





    async def load_calibration_data_from_rosbags(self):#, selected_camera:dict, selected_lidars:'list[dict]'):
        selected_rosbags = await self.rosbags_selector.get_selected_rows()
        selected_camera = await self.image_topics_select.get_selected_row()
        selected_lidars = await self.pcd_topics_select.get_selected_rows()
        init_check_err_msg = ''
        if len(selected_rosbags) == 0:
            init_check_err_msg = 'Please select one or more rosbags'
        elif selected_camera == None:
            init_check_err_msg = 'Please select one image topic'
        elif len(selected_lidars) == 0:
            init_check_err_msg = 'Please select one or more point cloud topics'
        if init_check_err_msg != '':
            self.show_dialog(init_check_err_msg)
            return

        print(selected_rosbags, selected_camera, selected_lidars, sep='\n')

        bags_loader = RosbagLoader(
            image_topics = [selected_camera],
            pcd_topics = selected_lidars,
            temp_dir = self.temp_dir,
            pcd_accumulation_frames = self.pcd_accumulation_frames
        )
        bags_loader.load_from_multiple_rosbag_files(
            bag_files = [bag['rosbag_file'] for bag in selected_rosbags],
            target_time = self.rosbag_target_time,
            save_dir = self.data_dir
        )




    async def load_calibration_data_from_single_rosbag(self, target_times:'list[float]'):
        selected_rosbags = await self.rosbags_selector.get_selected_rows()
        selected_camera = await self.image_topics_select.get_selected_row()
        selected_lidars = await self.pcd_topics_select.get_selected_rows()
        init_check_err_msg = ''
        if len(selected_rosbags) == 0:
            init_check_err_msg = 'Please select one rosbag'
        elif len(selected_rosbags) > 1:
            init_check_err_msg = 'Please select only one rosbag'
        elif selected_camera == None:
            init_check_err_msg = 'Please select one image topic'
        elif len(selected_lidars) == 0:
            init_check_err_msg = 'Please select one or more point cloud topics'
        if init_check_err_msg != '':
            self.show_dialog(init_check_err_msg)
            return
        selected_rosbag = selected_rosbags[0]
        bag_loader = RosbagLoader(
            image_topics = [selected_camera],
            pcd_topics = selected_lidars,
            temp_dir = self.temp_dir,
            pcd_accumulation_frames = self.pcd_accumulation_frames
        )
        bag_loader.load_from_single_rosbag_file(
            bag_file = selected_rosbag['rosbag_file'],
            target_times = target_times,
            save_dir = self.data_dir
        )




    def auto_detect_and_load_from_single_rosbag(self):
        # Show persistent dialog
        self.persistent_dialog = PersistentDialog(
            'Detecting valid 3D chessboards from rosbag',
            with_progress=True,
        )
        self.current_task = 'Detect valid 3D chessboards from rosbag'
        self.progress = 0.
        self.persistent_dialog.progress_bar.bind_value_from(self, 'progress')
        
        async def _worker():
            selected_rosbags = await self.rosbags_selector.get_selected_rows()
            selected_camera = await self.image_topics_select.get_selected_row()
            selected_lidars = await self.pcd_topics_select.get_selected_rows()
            init_check_err_msg = ''
            if len(selected_rosbags) == 0:
                init_check_err_msg = 'Please select one rosbag'
            elif len(selected_rosbags) > 1:
                init_check_err_msg = 'Please select only one rosbag'
            elif selected_camera == None:
                init_check_err_msg = 'Please inspect the rosbag then select one image topic'
            elif len(selected_lidars) != 1:
                init_check_err_msg = 'Please select only one point cloud topic'
            if init_check_err_msg != '':
                self.persistent_dialog.close()
                self.show_dialog(init_check_err_msg)
                return
            
            selected_rosbag = selected_rosbags[0]
            bag_loader = RosbagLoader(
                image_topics = [selected_camera],
                pcd_topics = selected_lidars,
                temp_dir = self.temp_dir,
                pcd_accumulation_frames = self.pcd_accumulation_frames
            )
            time_range = self.auto_detect_time_range._props['model-value']

            await bag_loader.auto_poses_finder(
                bag_file = selected_rosbag['rosbag_file'],
                image_topic = selected_camera['topic'],
                pcd_topic = selected_lidars[0]['topic'],
                ncols = self.horizontal_inner_corners+1,
                nrows = self.vertical_inner_corners+1,
                square_size = self.chessboard_square_size,
                save_dir = self.data_dir,
                start_time = time_range['min'],
                end_time = time_range['max'],
                time_interval = self.auto_detect_interval_length,
                queue = self.progress_value_queue
            )

            # Finish up
            self.persistent_dialog.close()
            self.show_dialog('Finished automatic loading from rosbag')

            # self.auto_poses_detection_area.clear()
            # self.auto_detected_poses_to_load = {}
            # with self.auto_poses_detection_area:
            #     for tt, img in bag_loader.collected_images_dirs.items():
            #         with ui.column():
            #             self.auto_detected_poses_to_load[tt] = ui.checkbox(
            #                 f'Rosbag elapsed time: {tt:.1f}',
            #                 value=True
            #             ).classes('w-full')
            #             ui.image(img).classes('w-full').props('fit=contain')
        
        ui.timer(0.001, _worker, once=True)





    def make_rosbag_load_tab(self):
        self.tab_panels.set_value('load_rosbag')
        rosbag_files = sorted(glob(os.path.join(self.data_dir, 'rosbags', '*.bag')))
        for dir in sorted(os.listdir(os.path.join(self.data_dir, 'rosbags'))):
            rosbag2_files = sorted(glob(os.path.join(self.data_dir, 'rosbags', dir, '*.db3')))
            if len(rosbag2_files) > 1:
                self.show_dialog('Rosbag 2 found but there should not be more than one .db3 file in a directory')
                return
            elif len(rosbag2_files) == 1:
                rosbag_files.append(rosbag2_files[0])

        if len(rosbag_files) == 0:
            self.show_dialog('No rosbag files (.bag, .db3) found on the data directory')
            return

        self.data_dir_input.disable()
        self.load_file_button.disable()
        self.load_rosbag_button.disable()
        self.edit_data_dir_button.enable()
        self.rosbag_load_tab.clear()
        self.rosbag_target_times = []

        columns = [
            {'headerName':'ID', 'field':'rosbag_id', 'checkboxSelection':True, 'headerCheckboxSelection':True, 'width':100, 'type':'rightAligned'},
            {'headerName':'Rosbag file name', 'field':'rosbag_file', 'flex':4},
            {'headerName':'N. topics', 'field':'num_topics', 'flex':1},
            {'headerName':'Duration', 'field':'duration', 'flex':1},
            {'headerName':'Modified time', 'field':'datetime', 'flex':2},
        ]
        rows = []

        for i, rosbag_file in enumerate(rosbag_files):
            if rosbag_file.endswith('.db3'):
                bag_dir = os.path.dirname(rosbag_file)
            elif rosbag_file.endswith('.bag'):
                bag_dir = rosbag_file
            with AnyReader([Path(bag_dir)]) as bag_reader:
                baginfo = bag_reader.connections
            bag_mtime = datetime.fromtimestamp(os.path.getmtime(rosbag_file))

            rows.append({
                'rosbag_id': i,
                'rosbag_file': rosbag_file,
                'num_topics': len(baginfo),
                #'filesize': f'{os.path.getsize(rosbag_file) / 1e9:.1f} GB',
                'duration': f'{bag_reader.duration / 1e9:.1f} s',
                'datetime': bag_mtime.strftime(r'%d %b %Y at %H:%M:%S')
            })

        image_list_columns = [
            {'headerName':'Camera name (editable)', 'field':'sensor-name', 'width':250, 'editable':True, 'checkboxSelection':True},
            {'headerName':'Image topic', 'field':'topic', 'flex':3},
            {'headerName':'Frequency', 'field':'freq', 'flex':1},
        ]

        pcd_list_columns = [
            {'headerName':'LiDAR name (editable)', 'field':'sensor-name', 'width':250, 'editable':True, 'checkboxSelection':True, 'headerCheckboxSelection':True},
            {'headerName':'Point cloud topic', 'field':'topic', 'flex':3},
            {'headerName':'Frequency', 'field':'freq', 'flex':1},
        ]

        def do_rosbag_inspection(rosbag_id:int, selected_cam:str, selected_lidar:str):
            try: self.image_topics_select
            except: return
            persistent_dialog = PersistentDialog('Loading rosbag data')

            rosbag_file = rosbag_files[rosbag_id]
            if rosbag_file.endswith('.db3'):
                bag_dir = os.path.dirname(rosbag_file)
            elif rosbag_file.endswith('.bag'):
                bag_dir = rosbag_file
            with AnyReader([Path(bag_dir)]) as bag_reader:
                baginfo = bag_reader.connections
                bag_duration = bag_reader.duration / 1e9

            lidar_name_hints = {
                'velodyne': 'vls128',
                'pandar': 'pandar',
                'iv_points': 'falcon',
                'livox': 'livox',
                'ouster128': 'os128',
                'ouster64': 'os64',
            }
            async def _worker(selected_cam, selected_lidar):
                image_topics = []
                pcd_topics = []
                for topic_data in baginfo:
                    topic = topic_data.topic
                    if topic_data.msgtype.lower().find('image') > -1:
                        if topic.find('flir_boson') > -1:
                            cam_prefix = 't'
                        elif topic.find('ladybug') > -1:
                            cam_prefix = 'rgb'
                        else:
                            cam_prefix = 'cam'
                        cam_num = re.search('[0-9]', topic)
                        if cam_num is not None:
                            cam_num = int(cam_num.group())
                        else:
                            cam_num = len(image_topics)
                        image_topics.append({
                            'sensor-name': f'{cam_prefix}_{cam_num}',
                            'topic': topic,
                            #'msg_count': topic_data[1],
                            'freq': f'{topic_data.msgcount / bag_duration:.2f} Hz'
                        })
                        #image_data.append(topic_data)
                    elif topic_data.msgtype == 'sensor_msgs/msg/PointCloud2':
                        lidar_name = ''
                        for hint, lidar in lidar_name_hints.items():
                            if topic.find(hint) > -1:
                                lidar_name = lidar
                                break
                        if lidar_name == '':
                            lidar_name = f'lidar_{len(pcd_topics)}'
                        pcd_topics.append({
                            'sensor-name': lidar_name,
                            'topic': topic,
                            #'msg_count': topic_data[1],
                            'freq': f'{topic_data.msgcount / bag_duration:.2f} Hz'
                        })
                        #pcd_data.append(topic_data)

                image_topics = sort_list_of_dicts_by_key(image_topics, 'sensor-name')
                pcd_topics = sort_list_of_dicts_by_key(pcd_topics, 'sensor-name')

                # Update list of image topics data
                #self.image_topics_select.rows[:] = image_topics
                self.image_topics_select.options['rowData'] = image_topics
                self.image_topics_select.update()

                # Update list of point cloud topics data
                self.pcd_topics_select.options['rowData'] = pcd_topics
                self.pcd_topics_select.update()
                self.pcd_topics_select.call_api_method('selectAll')

                # Load rosbag
                bag_loader = RosbagLoader(
                    image_topics, pcd_topics, self.temp_dir,
                    pcd_accumulation_frames = self.pcd_accumulation_frames
                )
                image_data, pcd_data = await bag_loader._single_rosbag_loader(
                    rosbag_file, target_times=[self.rosbag_target_time],
                )

                # Update camera and LiDAR names
                self.select_inspect_cam.set_options(bag_loader.cam_names)
                self.select_inspect_lidar.set_options(bag_loader.lidar_names)
                self.select_inspect_cam.update()
                self.select_inspect_lidar.update()

                # Display image and point cloud data
                self.rosbag_inspection_area.clear()
                with self.rosbag_inspection_area:
                    if not isinstance(selected_cam, str):
                        selected_cam = bag_loader.cam_names[0]
                        self.select_inspect_cam.set_value(selected_cam)
                    for cam_name in bag_loader.cam_names:
                        if cam_name != selected_cam: continue
                        with ui.image(image_data[self.rosbag_target_time][cam_name]).classes('w-full').props('fit=contain'):
                            ui.label(f'{cam_name}').classes('relative-top text-subtitle2 text-center')

                self.rosbag_pcd_inspection_area.clear()
                if not isinstance(selected_lidar, str):
                    selected_lidar = bag_loader.lidar_names[0]
                    self.select_inspect_lidar.set_value(selected_lidar)
                with self.rosbag_pcd_inspection_area:
                    self.show_point_cloud(pcd_data[self.rosbag_target_time][selected_lidar])

                await asyncio.sleep(0.01)
                persistent_dialog.close()

            ui.timer(0.1, lambda:_worker(selected_cam, selected_lidar), once=True)


        def add_rosbag_time():
            if self.rosbag_target_time not in self.rosbag_target_times:
                self.rosbag_target_times.append(self.rosbag_target_time)
                rosbag_target_times_selector.set_options(self.rosbag_target_times)
                rosbag_target_times_selector.set_value(self.rosbag_target_times)
                rosbag_target_times_selector.update()


        with self.rosbag_load_tab:
           # ui.label('Select rosbag files to load data from:').style('font-weight: bold')
            with ui.grid(columns=2):
                with ui.grid(columns=1).classes('items-stretch').style(r'height: 620px'):
                    self.image_topics_select = ui.aggrid({
                        'defaultColDef': {'resizable':True, 'sortable':True},
                        'columnDefs': image_list_columns,
                        'rowData': [],
                        'rowSelection': 'single',
                        'rowMultiSelectWithClick': True,
                        'animateRows': True
                    }, theme='alpine-dark').classes('w-full').style(r'height: 300px')

                    self.pcd_topics_select = ui.aggrid({
                        'defaultColDef': {'resizable':True, 'sortable':True},
                        'columnDefs': pcd_list_columns,
                        'rowData': [],
                        'rowSelection': 'multiple',
                        'rowMultiSelectWithClick': True,
                        'animateRows': True
                    }, theme='alpine-dark').classes('w-full').style(r'height: 300px')

                self.rosbags_selector = ui.aggrid({
                    'defaultColDef': {'resizable':True, 'sortable':True},
                    'columnDefs': columns,
                    'rowData': rows,
                    'rowSelection': 'multiple',
                    'rowMultiSelectWithClick': True,
                    'animateRows': True
                },theme='alpine-dark').classes('w-full').style(r'height: 620px')
                self.rosbags_selector.call_api_method('selectAll')

                do_rosbag_inspection(0, None, None)

            ui.html("<p>&nbsp<p>")

            with ui.grid(columns=2):
                def handle_inspect_rosbag_change():
                    new_rosbag_duration = float(rows[self.single_rosbag_id]['duration'].rstrip(' s'))
                    rosbag_play_progress.props('max', new_rosbag_duration)
                    rosbag_play_progress.update()
                    self.auto_detect_time_range._props['max'] = new_rosbag_duration
                    self.auto_detect_time_range.update()

                inspect_rosbag_selector = ui.select(
                    label='Select rosbag to inspect',
                    options={i: rosbag_files[i] for i in range(len(rosbag_files))},
                    value=0,
                    on_change=handle_inspect_rosbag_change
                ).classes('w-full').props('filled').bind_value_to(self, 'single_rosbag_id')

                ui.button(
                    'Load selected multiple rosbags',
                    on_click=lambda: self.load_calibration_data_from_rosbags()
                ).classes('w-full')

            ui.html("<p>&nbsp<p>")
            ui.separator()
            ui.html("<p>&nbsp<p>")

            ui.label('Single rosbag viewer').style('font-weight: bold')
            ui.html("<p>&nbsp<p>")
            with ui.grid(columns=2).classes('w-full'):
                self.select_inspect_cam = ui.select(
                    label='Inspected camera',
                    options=[],
                    on_change=lambda x: do_rosbag_inspection(inspect_rosbag_selector.value, x.value, self.select_inspect_lidar.value)
                ).classes('w-full').props('filled')

                with ui.grid(columns=2).classes('w-full'):
                    self.select_inspect_lidar = ui.select(
                        label='Inspected LiDAR',
                        options=[],
                        on_change=lambda x: do_rosbag_inspection(inspect_rosbag_selector.value, self.select_inspect_cam.value, x.value)
                    ).classes('w-full').props('filled')
                    pcds_accu_frames_slider = ui.slider(
                        min=0, max=10, step=1, value=0,
                        on_change=lambda: pcds_accu_frames_slider.props(f':label-value="\'Num pcd frames to accumulate: {pcds_accu_frames_slider.value}\'"')
                    ).props(r'label dense label-always snap switch-label-side').bind_value_to(self, 'pcd_accumulation_frames')
                    pcds_accu_frames_slider.props(f':label-value="\'Num pcd frames to accumulate: {pcds_accu_frames_slider.value}\'"')
                self.rosbag_inspection_area = ui.grid(columns=1).classes('w-full')
                self.rosbag_pcd_inspection_area = ui.grid(columns=1).classes('w-full')

            ui.html("<p>&nbsp<p>")
            rosbag_play_progress = ui.slider(
                min = 0.,
                max = float(rows[self.single_rosbag_id]['duration'].rstrip(' s')),
                step = 0.1,
            ).props('snap').bind_value(self, 'rosbag_target_time')

            with ui.grid(columns=2).classes('w-full'):
                with ui.grid(columns=3).classes('w-full'):
                    rosbag_target_time_selector = ui.number(
                        label='Target rosbag elapsed time',
                        value=2, step=0.1, suffix='s'
                    ).props(r'filled input-class="text-right"').bind_value(self, 'rosbag_target_time')

                    ui.button(
                        text='Inspect',
                        on_click=lambda: do_rosbag_inspection(
                            inspect_rosbag_selector.value,
                            self.select_inspect_cam.value,
                            self.select_inspect_lidar.value
                        )
                    ).classes('w-full')

                    ui.button(
                        'Add rosbag time',
                        on_click=add_rosbag_time
                    ).classes('w-full')

                with ui.grid(columns=2).classes('w-full'):
                    rosbag_target_times_selector = ui.select(
                        options=[], label='Selected target times', multiple=True
                    ).classes('w-full').props('filled')

                    ui.button(
                        'Load single rosbag',
                        on_click=lambda: self.load_calibration_data_from_single_rosbag(
                            target_times = rosbag_target_times_selector.value
                        )
                    )
            
            ui.html("<p>&nbsp<p>")
            ui.label('Automatic detection and loading from rosbag').style('font-weight: bold')
            ui.html("<p>&nbsp<p>")
            
            with ui.grid(columns=2).classes('w-full'):
                bag_duration = float(rows[self.single_rosbag_id]['duration'].rstrip(' s'))
                with ui.column():
                    ui.html("<p>&nbsp<p>")
                    self.auto_detect_time_range = ui.element('q-range').classes('w-full')
                self.auto_detect_time_range._props = {
                    'min': 0,
                    'max': bag_duration,
                    'step': 1,
                    'model-value': {'min': 10., 'max': bag_duration - 10.},
                    'snap': True,
                    'label-always': True,
                }
                def handle_change(e):
                    self.auto_detect_time_range._props['model-value'] = e.args
                    self.auto_detect_time_range.update()
                self.auto_detect_time_range.on('update:model-value', handle_change)

                with ui.grid(columns=2).classes('w-full'):
                    ui.number(
                        label = 'Interval length',
                        value = 1.,
                        min = 0.1,
                        step = 0.1,
                        suffix = 's'
                    ).props('filled input-class="text-right"').bind_value_to(self, 'auto_detect_interval_length')
                    ui.button(
                        'Auto detect poses',
                        on_click=lambda: self.auto_detect_and_load_from_single_rosbag()
                    ).classes('w-full')
                    # ui.button(
                    #     'Load auto-detected poses',
                    #     on_click=lambda: self.load_calibration_data_from_single_rosbag(
                    #         target_times = 
                    #     )
                    # )
            
            self.auto_poses_detection_area = ui.grid(columns=4).classes('w-full')





    def show_point_cloud(
            self,
            pcd:'o3d.geometry.PointCloud | np.ndarray'=None,
            chessboard_poses:'list[np.ndarray]'=[],
            all_chessboard_3d_corners:'list[np.ndarray]'=[],
            height:int=600,
            cmap='winter',
            colour_attribute='intensity',
            point_size:float=0.02
    ):
        if pcd is not None:
            if isinstance(pcd, np.ndarray):
                points = pcd[:,:3]
                if colour_attribute == 'intensity':
                    I = pcd[:,3]
                    I /= np.max(I)
                    colors = plt.get_cmap(cmap)(I)[:, 0:3]
                elif colour_attribute == 'RGB':
                    colors = pcd[:, 4:7]
            else:
                points = np.array(pcd.points)
                colors = np.array(pcd.colors)
        camera_adjusted = False

        with ui.scene(height=height, grid=False).classes('w-full') as pcd_display:
            # Chessboard bounding boxes
            if len(chessboard_poses) > 0:
                for i in range(len(chessboard_poses)):
                    chessboard_pose = chessboard_poses[i]
                    if not isinstance(chessboard_pose, np.ndarray): continue
                    chessboard = pcd_display.box(
                        width = (self.horizontal_inner_corners + 1) * self.chessboard_square_size,
                        height = (self.vertical_inner_corners + 1) * self.chessboard_square_size,
                        depth = 0.04,
                        wireframe = True
                    )
                    chessboard.color = '#FF0000'
                    chessboard.move(x=chessboard_pose[0,3], y=chessboard_pose[1,3], z=chessboard_pose[2,3])
                    chessboard.rotate_R(chessboard_pose[:3,:3].tolist())

                    # Pose id indicator
                    pose_id_label = pcd_display.text3d(
                        f'Pose {i}',
                        f'color: rgba(1, 0, 0, 1); background: rgba(0, 0, 0, 0.2); border-radius: 5px; padding: 5px'
                    )
                    pose_id_label.rotate_R(chessboard_pose[:3, :3].tolist())
                    pose_id_label.move(x=chessboard_pose[0, 3], y=chessboard_pose[1, 3], z=chessboard_pose[2, 3])
                    pose_id_label.scale(0.005)

                    # Adjust camera angle
                    if not camera_adjusted:
                        pcd_display.move_camera(
                            x = 0., y = 0., z = 0.2,
                            look_at_x = chessboard_pose[0],
                            look_at_y = chessboard_pose[1],
                            look_at_z = chessboard_pose[2],
                        )
                        camera_adjusted = True

            # 3D corners
            for i in range(len(all_chessboard_3d_corners)):
                chessboard_3d_corners = all_chessboard_3d_corners[i]
                if not isinstance(chessboard_3d_corners, np.ndarray): continue
                try:
                    corners_3D_template
                except NameError:
                    corners_3D_template = np.tile(np.array([1,0,0]), (chessboard_3d_corners.shape[0], 1))
                    corners_3D_template[0] = np.array([1,1,0])
                    corners_3D_template[-1] = np.array([0,1,0])
                try:
                    points = np.vstack((points, chessboard_3d_corners))
                    colors = np.vstack((colors, corners_3D_template))
                except NameError:
                    points = chessboard_3d_corners
                    colors = corners_3D_template

            # Point cloud
            pcd_display.point_cloud(points=points.tolist(), colors=colors.tolist(), point_size=point_size)





    def show_dialog(self, msg:str, width:int=400, height:int=150):
        with ui.dialog(value=True) as dialog, ui.card().style(f'width: {width}px; height: {height}px'):
            with ui.column().classes('absolute-center items-center w-full'):
                ui.label(text=msg).style(f'width:{width-100}px;')
                ui.button('OK', on_click=dialog.close)



    def show_persistent_dialog(self, msg:str, width:int=400, height:int=200):
        with ui.dialog(value=True) as self.persistent_dialog, ui.card().style(f'width: {width}px; height: {height}px'):
            self.persistent_dialog.props('persistent')
            with ui.column().classes('absolute-center items-center w-full'):
                ui.spinner(size='3em')
                ui.label(text=msg)
                self.persistent_dialog.open()
    




    def make_sidebar(self):
        def do_edit_dir():
            self.data_dir_input.enable()
            self.load_file_button.enable()
            self.load_rosbag_button.enable()
            self.edit_data_dir_button.disable()

        with self.sidebar:
            init_parent_dir = os.path.dirname(self.data_dir)
            self.data_dir_input = ui.select(
                options=[os.path.join(init_parent_dir, dir) for dir in sorted(os.listdir(init_parent_dir))],
                label='Data directory',
                value=self.data_dir, with_input=True,
            ).bind_value_to(self, 'data_dir').props('filled')
            def data_dir_input_on_type(e):
                current_data_dir = e.args[0]
                if current_data_dir == '': return
                parent_dir = os.path.dirname(current_data_dir)
                try:
                    children_dirs = sorted(os.listdir(parent_dir))
                    suggestions = [os.path.join(parent_dir, dir) for dir in children_dirs]
                    self.data_dir_input.set_options(suggestions)
                except:
                    return
            self.data_dir_input.on('filter', lambda e: data_dir_input_on_type(e))
            
            ui.html("<p>&nbsp</p>")
            self.chessboard_configs_expansion = ui.expansion(
                'Chessboard target dimensions', icon='settings', value=True
            ).props('header-style="font-weight: bold"')
            with self.chessboard_configs_expansion:
                ui.html("<p>&nbsp</p>"); ui.html("<p>&nbsp</p>")
                with ui.grid(columns=2):
                    self.horizontal_slider = ui.slider(
                        min=2,
                        max=16,
                        value=8,
                        on_change=lambda: self.horizontal_slider.props(f':label-value="\'Horizontal: {self.horizontal_slider.value}\'"')
                    ).props(r'label dense label-always snap')
                    self.horizontal_slider.props(f':label-value="\'Horizontal: {self.horizontal_slider.value}\'"')
                    self.horizontal_slider.bind_value_to(self, 'horizontal_inner_corners', lambda x: x-1)
                    self.horizontal_slider.bind_enabled_from(self, 'calibration_session_started', lambda x: not x)

                    self.vertical_slider = ui.slider(
                        min=2,
                        max=10,
                        value=6,
                        on_change=lambda: self.vertical_slider.props(f':label-value="\'Vertical: {self.vertical_slider.value}\'"')
                        ).props(r'label dense label-always snap')
                    self.vertical_slider.props(f':label-value="\'Vertical: {self.vertical_slider.value}\'"')
                    self.vertical_slider.bind_value_to(self, 'vertical_inner_corners', lambda x: x-1)
                    self.vertical_slider.bind_enabled_from(self, 'calibration_session_started', lambda x: not x)

                    camera_model_selector = ui.select(
                        ['pinhole', 'fisheye'], label='Camera model'
                    ).props('filled hide-dropdown-icon').bind_value(self, 'camera_model')
                    camera_model_selector.bind_enabled_from(self, 'calibration_session_started', lambda x: not x)

                    square_size_selector = ui.number(
                        label='Square size',
                        min=1, max=20, step=0.1,
                        value=self.chessboard_square_size*100,
                        suffix='cm'
                    ).props(r'filled input-class="text-right"')
                    square_size_selector.bind_value_to(self, 'chessboard_square_size', lambda x:x/100)
                    square_size_selector.bind_enabled_from(self, 'calibration_session_started', lambda x: not x)

                ui.html("<p>&nbsp</p>"); ui.separator()

            def cancel_calibration():
                del self.calibrator
                self.calibrator:Calibrator = None
                del self.sensors_data
                self.sensors_data:DataContainer = None
                self.calibration_session_started = False
                self.calibrated = False
                self.chessboard_configs_expansion.set_value(True)
                self.edit_data_dir_button.enable()
                self.calibrate_intrinsic_tab_created = False
                self.created_3d_corners_detection_tab = False
                self.LiDAR_camera_calibration_tab_created = False
                for stage in self.calibration_stages.__dict__.keys():
                    self.calibration_stages.__setattr__(stage, False)
                self.stepper.set_value('Load data')

            # ui.html("<p>&nbsp</p>")
            # with ui.grid(columns=2):
            #     self.save_calibration_button = ui.button('Save').bind_enabled_from(self, 'calibrated')

            ui.html("<p>&nbsp</p>")
            ui.label('Calibration pipeline').style('font-weight: bold')
            ui.html("<p>&nbsp</p>")
            with ui.stepper().props('vertical header-nav animated').classes('w-full') as self.stepper:
                with ui.step('Load data'):
                    ui.label('Load data from:')
                    with ui.grid(columns=2).classes('w-full'):
                        self.load_file_button = ui.button('File', on_click=lambda:self.make_load_data_from_file_tab())
                        self.load_rosbag_button = ui.button('Rosbags', on_click=lambda:self.make_rosbag_load_tab())
                    ui.html("<p>&nbsp</p>")
                    self.edit_data_dir_button = ui.button('Edit directory', on_click=do_edit_dir).classes('w-full')
                    self.edit_data_dir_button.disable()

                    ui.html("<p>&nbsp</p>")
                    ui.label('Calibration session:')
                    with ui.grid(columns=2).classes('w-full'):
                        self.start_calibration_button = ui.button(
                            'Start', on_click=lambda: self.load_sensors_data()
                        ).classes('w-full').bind_enabled_from(self, 'calibration_session_started', lambda x: not x)
                        self.cancel_calibration_button = ui.button(
                            'End', on_click=cancel_calibration
                        ).bind_enabled_from(self, 'calibration_session_started')

                with ui.step('Calibrate camera'):
                    ui.label('Calibrate camera intrinsic parameters')
                    with ui.grid(columns=2).classes('w-full'):
                        ui.button('Load', on_click=lambda: self.load_camera_intrinsics())
                        ui.button('Calibrate', on_click=lambda: self.show_dialog('Feature not yet implemented'))
                    
                    def update_raw_images_already_undistorted():
                        self.calibrator.raw_images_already_undistorted = self.raw_images_already_undistorted
                        self.calibrator.cam_intrinsics.raw_images_already_undistorted = self.raw_images_already_undistorted
                    ui.html("<p>&nbsp</p>")
                    ui.switch(
                        'Raw images already undistorted',
                        on_change=update_raw_images_already_undistorted
                    ).bind_value(self, 'raw_images_already_undistorted')
                    ui.html("<p>&nbsp</p>")
                    
                    with ui.grid(columns=2).classes('w-full'):
                        ui.button(
                            'Next', on_click=lambda: self.stepper.next()
                        ).bind_enabled_from(self.calibration_stages, 'calibrate_intrinsics')
                        ui.button('Back', on_click=self.stepper.previous).props('flat')

                with ui.step('Detect 3D corners'):
                    ui.label('ROI:').style('font-weight: bold')
                    ui.html("<p>&nbsp</p>"); ui.html("<p>&nbsp</p>")
                    with ui.grid(columns=2).classes('w-full'):
                        self.roi_direction_slider = ui.slider(
                            min=-170, max=180, step=10, value=0,
                            on_change=lambda: self.roi_direction_slider.props(f':label-value="\'Direction: {self.roi_direction_slider.value}°\'"')
                        ).props(r'label label-always snap')
                        self.roi_direction_slider.props(f':label-value="\'Direction: {self.roi_direction_slider.value}°\'"')

                        self.roi_width_slider = ui.slider(
                            min=0, max=180, step=10, value=30,
                            on_change=lambda: self.roi_width_slider.props(f':label-value="\'Width: {self.roi_width_slider.value}°\'"')
                        ).props(r'label label-always snap')
                        self.roi_width_slider.props(f':label-value="\'Width: {self.roi_width_slider.value}°\'"')

                    ui.html("<p>&nbsp</p>")
                    ui.select(
                        label='3D target detector',
                        options={0: 'LiDAR image-based', 1: 'Plane-based', 2: 'Hybrid'},
                        value=2,
                    ).props('filled').bind_value(self, 'corners_3D_detector')
                    ui.html("<p>&nbsp</p>")
                    self.LiDAR_name_selector = ui.select(
                        label='Select LiDAR sensor',
                        options=[],
                    ).props('filled').classes('w-full').bind_value_to(self, 'selected_LiDAR')
                    ui.html("<p>&nbsp</p>")
                    ui.button('Detect', on_click=lambda: self.detect_3d_corners()).classes('w-full')
                    ui.html("<p>&nbsp</p>")
                    with ui.grid(columns=2).classes('w-full'):
                        ui.button('Next',
                            on_click=self.stepper.next
                        ).bind_enabled_from(self.calibration_stages, 'detect_3d_corners')
                        ui.button('Back', on_click=self.stepper.previous).props('flat')

                with ui.step('Detect 2D corners'):
                    self.bio_retina_switch = ui.switch('Use bio-retina enhancement')
                    ui.html("<p>&nbsp</p>")
                    ui.button('Detect',
                        on_click=lambda: self.detect_2d_corners()
                    ).classes('w-full')
                    ui.html("<p>&nbsp</p>")
                    with ui.grid(columns=2).classes('w-full'):
                        ui.button('Next',
                            on_click=self.stepper.next
                        ).bind_enabled_from(self.calibration_stages, 'detect_2d_corners')
                        ui.button('Back', on_click=self.stepper.previous).props('flat')

                with ui.step('Calibrate LiDAR-camera'):
                    sensor_fusion_config_expansion = ui.expansion(
                        'Fusion configs', icon='settings', value=False
                    ).props('header-style="font-weight: bold"').bind_enabled_from(self, 'calibrated')
                    with sensor_fusion_config_expansion:
                        self.sensor_fusion_attribute_selector = ui.select(
                            label='Fuse colour by',
                            options=['depth', 'intensity', 'z-coordinates'],
                            on_change=lambda: self.update_LiDAR_to_cam_fusion_images()
                        ).props('filled').bind_value(self, 'sensor_fusion_attribute')
                        ui.html("<p>&nbsp</p>"); ui.html("<p>&nbsp</p>")
                        self.max_fusion_depth_slider = ui.slider(
                            min=1, max=100, step=1, value=15,
                            on_change=lambda: self.max_fusion_depth_slider.props(f':label-value="\'Max fusion depth: {self.max_fusion_depth_slider.value}m\'"')
                        ).props(r'label label-always snap')
                        self.max_fusion_depth_slider.props(f':label-value="\'Max fusion depth: {self.max_fusion_depth_slider.value}m\'"')
                        self.max_fusion_depth_slider.bind_value_to(self, 'max_fusion_depth')
                        self.max_fusion_depth_slider.on(
                            'change', lambda: self.update_LiDAR_to_cam_fusion_images()
                        )
                    ui.html("<p>&nbsp</p>")                    
                    ui.button('Calibrate',
                        on_click=lambda: self.calibrate_LiDAR_camera()
                    ).classes('w-full').bind_text_from(self, 'calibrated', lambda x: 'Re-calibrate' if x else 'Calibrate')

            def stepper_transition(event):
                new_val = event.args[0]
                old_val = event.args[1]
                if old_val == 'Load data' and not self.calibration_session_started:
                    self.show_dialog('Please load data and start calibration session first')
                    self.stepper.set_value('Load data')
                    return
                if new_val == 'Load data':
                    self.tab_panels.set_value('load_data')
                if new_val == 'Calibrate camera':
                    self.make_calibrate_intrinsics_tab()
                elif new_val == 'Detect 3D corners':
                    self.make_3d_corners_detection_tab()
                elif new_val == 'Detect 2D corners':
                    self.make_2d_corners_detection_tab()
                elif new_val == 'Calibrate LiDAR-camera':
                    if self.calibrator is None or len(self.calibrator.valid_LiDAR_poses) == 0:
                        self.show_dialog('Please detect 3D chessboard corners first')
                        self.stepper.set_value(old_val)
                        return
                    else: self.make_LiDAR_camera_calibration_tab()
            self.stepper.on('transition', stepper_transition)



    def gui(self):
        with ui.header().classes(replace='row items-center') as header:
            with ui.row().classes('items-center'):
                ui.button(on_click=lambda: self.sidebar.toggle(), icon='menu').props('flat color=white')
                ui.label('Current task:')
                ui.label('None').bind_text_from(self, 'current_task')
            self.progress_bar = ui.linear_progress(
                show_value=False, color='#FEBE10', size=10
            ).bind_value_from(self, 'progress')

        ui.timer(0.01, self.update_progress)

        with ui.tabs() as tabs:
            ui.tab('load_data', label='Load data from file')
            ui.tab('load_rosbag', label='Load data from rosbag')
            ui.tab('cam_calib', label='Camera intrinsic calibrator')
            ui.tab('3D_chessboard', label='3D corners detector')
            ui.tab('2D_chessboard', label='2D corners detector')
            ui.tab('LiDAR_cam_calib', label='LiDAR-camera calibrator')
            ui.tab('lidars_extrinsic', label='LiDARs extrinsic refinement')
            ui.tab('eval', label='Evaluation')
            ui.tab('fusion', label='Sensors fusion')
        self.tab_panels = ui.tab_panels(tabs, value='load_data').classes('w-full')

        def tab_transition(vals):
            newval, oldval = vals.args
            #if newval == '3D_chessboard': self.make_3d_corners_detection_tab()
        self.tab_panels.on('transition', tab_transition)

        with self.tab_panels:
            self.load_file_tab = ui.tab_panel('load_data')
            self.rosbag_load_tab = ui.tab_panel('load_rosbag')
            self.calibrate_intrinsic_tab = ui.tab_panel('cam_calib')
            self.detect_3d_corners_tab = ui.tab_panel('3D_chessboard')
            self.detect_2d_corners_tab = ui.tab_panel('2D_chessboard')
            self.LiDAR_cam_calib_tab = ui.tab_panel('LiDAR_cam_calib')



    async def update_progress(self):
        if self.progress_value_queue.empty(): return
        self.progress = await self.progress_value_queue.get()







gui = CalibrationGUI()

### START SERVER ###
async def disconnect() -> None:
    """Disconnect all clients from current running server."""
    for client in nicegui.globals.clients.keys():
        await app.sio.disconnect(client)

def handle_sigint(signum, frame) -> None:
    # `disconnect` is async, so it must be called from the event loop; we use `ui.timer` to do so.
    ui.timer(0.1, disconnect, once=True)
    # Delay the default handler to allow the disconnect to complete.
    ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)

async def cleanup() -> None:
    try: shutil.rmtree('/tmp/calibrator_gui')
    except: pass
    await disconnect()
    #process_pool_executor.shutdown()

app.on_shutdown(cleanup)
signal.signal(signal.SIGINT, handle_sigint)

ui.run(title='LiDAR - Camera Calibration')
