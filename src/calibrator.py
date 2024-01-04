import os, shutil, sys, time, io
import asyncio
import multiprocessing as mp
import requests, base64
import numpy as np
import open3d as o3d
import pypcd4
import uuid
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
from PIL import Image
import cv2
import json, yaml
import scipy
import scipy.optimize
from scipy.spatial.transform import Rotation

from .utils.funcs import *
from .data_container import DataContainer



class CameraIntrinsics:
    def __init__(self, from_json:str=None, temp_dir:str=None, raw_images_already_undistorted:bool=False) -> None:
        # Intrinsics parameters
        self.FocalLength = np.array([0., 0.])
        self.PrinciplePoint = np.array([0., 0.])
        self.Skew = 0.
        self.RadialDistortion = np.zeros(3)
        self.TangentialDistortion = np.zeros(2)
        self.DistCoeffs = np.zeros(5)
        self.ImageSize = np.array([0, 0])
        self.IntrinsicMatrix = np.identity(3)
        self.raw_images_already_undistorted = raw_images_already_undistorted

        # Other parameters
        self.temp_dir = temp_dir
        self.mapx:float = None
        self.mapy:float = None

        if from_json is not None:
            self.load_json(from_json)


    def calibrate_intrinsics(self,
            images:'list[np.ndarray]',
            chessboard_size:'tuple[int, int]',
            chessboard_square_size:float,
    ):
        raise NotImplementedError
        corners = []
        for img in images:
            pass


    def undistort_image(self, img:np.ndarray, demo=False):
        if not self.raw_images_already_undistorted:
            undst_img = cv2.undistort(
                img, self.IntrinsicMatrix, self.DistCoeffs,
            )
        else:
            undst_img = img
        if demo:
            self.demo_undistort = {
                'raw': os.path.join(self.temp_dir, f'{uuid.uuid4()}.png'),
                'undistorted': os.path.join(self.temp_dir, f'{uuid.uuid4()}.png')
            }
            os.makedirs(self.temp_dir, exist_ok=True)
            if img.ndim == 2:
                plt.imsave(self.demo_undistort['raw'], img, cmap='gray')
                plt.imsave(self.demo_undistort['undistorted'], undst_img, cmap='gray')
            elif img.ndim == 3:
                plt.imsave(self.demo_undistort['raw'], img)
                plt.imsave(self.demo_undistort['undistorted'], undst_img)
        return undst_img


    def undistort_points(self, points:'np.ndarray'):
        undst_points = cv2.undistortPoints(
            points, self.IntrinsicMatrix, self.DistCoeffs,
            R=self.R, P=self.P
        )
        undst_points = np.array(undst_points, np.float32).reshape((-1,2))
        return undst_points


    def load_json(self, json_dir:str):
        if not os.path.isfile(json_dir):
            raise FileNotFoundError(f'Intrinsic json file {json_dir} does not exist')
        with open(json_dir, 'r') as f:
            json_data = json.load(f)
        self.FocalLength = np.array(json_data['FocalLength']).flatten()
        self.PrinciplePoint = np.array(json_data['PrinciplePoint']).flatten()
        self.Skew = json_data['Skew']
        self.RadialDistortion = np.array(json_data['RadialDistortion']).flatten()
        self.TangentialDistortion = np.array(json_data['TangentialDistortion']).flatten()
        self.ImageSize = np.array(json_data['ImageSize']).flatten()
        try:
            self.IntrinsicMatrix = np.array(json_data['IntrinsicMatrix'])
        except:
            self.IntrinsicMatrix = np.array([
                [self.FocalLength[0], self.Skew, self.PrinciplePoint[0]],
                [0., self.FocalLength[1], self.PrinciplePoint[1]],
                [0., 0., 1.]
            ])

        k1, k2, k3 = self.RadialDistortion
        p1, p2 = self.TangentialDistortion
        self.DistCoeffs = np.array([k1, k2, p1, p2, k3])
        self.R = np.identity(3, np.float32)
        self.P = np.zeros((3,4), np.float32)
        self.P[:3,:3] = self.IntrinsicMatrix


    def load_yaml(self, yaml_dir:str):
        if not os.path.isfile(yaml_dir):
            raise FileNotFoundError(f'Intrinsic yaml file {yaml_dir} does not exist')
        with open(yaml_dir, 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        self.ImageSize = np.array([yaml_data['image_width'], yaml_data['image_height']])
        self.IntrinsicMatrix = np.array(yaml_data['camera_matrix']['data']).reshape((3,3))
        self.FocalLength = np.array([self.IntrinsicMatrix[0,0], self.IntrinsicMatrix[1,1]])
        self.PrinciplePoint = np.array([self.IntrinsicMatrix[0,2], self.IntrinsicMatrix[1,2]])
        self.Skew = self.IntrinsicMatrix[0,1]
        self.DistCoeffs = np.array(yaml_data['distortion_coefficients']['data']).flatten()
        self.RadialDistortion = self.DistCoeffs[[0, 1, 4]]
        self.TangentialDistortion = self.DistCoeffs[[2, 3]]

        self.R = np.identity(3, np.float32)
        self.P = np.zeros((3,4), np.float32)
        self.P[:3,:3] = self.IntrinsicMatrix


    def save_json(self, json_dir:str):
        json_data = {
            'FocalLength': self.FocalLength.tolist(),
            'PrinciplePoint': self.PrinciplePoint.tolist(),
            'Skew': self.Skew,
            'RadialDistortion': self.RadialDistortion.tolist(),
            'TangentialDistortion': self.TangentialDistortion.tolist(),
            'ImageSize': self.ImageSize.tolist(),
            'IntrinsicMatrix': self.IntrinsicMatrix.tolist()
        }
        with open(json_dir, 'w') as f:
            json.dump(json_data, f, indent=4)




class Calibrator:
    def __init__(self,
            data:DataContainer,
            horizontal_inner_corners:int,
            vertical_inner_corners:int,
            chessboard_square_size:float,
            temp_dir:str,
            raw_images_already_undistorted:bool,
            cam_intrinsics:CameraIntrinsics=None,
    ) -> None:

        # Corners detection engines
        self.matlab_engine = None
        self.use_matlab = True
        self.corners_detector = None
        self.raw_images_already_undistorted = raw_images_already_undistorted

        # Calibration parameters
        self.use_bio_retina = False

        self.data = data
        if cam_intrinsics is None:
            self.cam_intrinsics = CameraIntrinsics(
                temp_dir=temp_dir,
                raw_images_already_undistorted=raw_images_already_undistorted
            )
        else:
            self.cam_intrinsics = cam_intrinsics

        # print('Temp dir exist =', os.path.exists(temp_dir), temp_dir)
        self.temp_dir = '/tmp/calibrator_gui'
        # self.temp_dir = os.path.join(temp_dir, 'corners_3D')
        # os.makedirs(os.path.join(self.temp_dir, 'corners_3D'), exist_ok=True)

        self.num_poses = len(data.images)
        self.valid_LiDAR_poses = []
        self.valid_cam_poses = []
        self.valid_poses = []

        # Corners detection results
        self.LiDAR_name:str = None
        self.all_chessboard_poses:'list[np.ndarray]' = [None for i in range(self.num_poses)]
        self.all_chessboard_3d_corners:'list[np.ndarray]' = [None for i in range(self.num_poses)]
        self.all_chessboard_2d_corners:'list[np.ndarray]' = [None for i in range(self.num_poses)]
        self.point_clouds_o3d = {lidar_name: [] for lidar_name in data.lidar_names}
        self.undistorted_images:'list[np.ndarray]' = [None for i in range(self.num_poses)]
        self.corners_3D_costs:'dict[int, float]' = {i: None for i in range(self.num_poses)}
        self.roi_direction = 0.
        self.roi_angular_width = 30.

        # LiDAR-camera pose optimisation
        self.opt_cam_pose:np.ndarray = None
        self.PnP_reprojection_errors:'list[float]' = [None for i in range(self.num_poses)]
        self.PnP_reprojection_images:'list[str]' = [None for i in range(self.num_poses)]
        self.fusion_images:'list[str]' = [None for i in range(self.num_poses)]

        # Configurations
        self.grid_inner_size = np.array([horizontal_inner_corners-1, vertical_inner_corners-1])
        self.chessboard_square_size = chessboard_square_size
        self.showcase = {'LiDAR_image': '', 'before_opt': '', 'after_opt': ''}
        self.opt_result_vis = [None for _ in range(self.num_poses)]
        self.corners_2d_demo_dir:'list[str]' = [None for _ in range(self.num_poses)]



    def detect_corners(self,
            img:np.ndarray,
            min_corner_metric:float=0.01,
            use_matlab=False,
            use_gpu=True,
            refine=False,
            undistort=False,
            demo=False,
            pose_id:int=None
    ):
        if use_matlab:
            if self.matlab_engine is None:
                import matlab
                import matlab.engine
                _engine_starter = matlab.engine.start_matlab(background=True)
                self.matlab_engine = _engine_starter.result(timeout=10)
            corners, board_size = detect_corners_matlab(
                img,
                min_corner_metric=min_corner_metric,
                matlab_engine=self.matlab_engine
            )
        else:
            raise NotImplementedError

        result_corners = np.array(corners)
        if undistort and not self.raw_images_already_undistorted and self.cam_intrinsics.ImageSize[0] > 0:
            img = self.cam_intrinsics.undistort_image(img)
            corners = self.cam_intrinsics.undistort_points(corners)

        if len(corners) > 0 and demo and pose_id is not None:
            fig, ax = plt.subplots(figsize=(12,8))
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.scatter(corners[1:-1,0], corners[1:-1,1], s=5, marker='x', c='r')
            ax.scatter(corners[0,0], corners[0,1], s=5, marker='s', facecolors='none', edgecolors='y')
            ax.scatter(corners[-1,0], corners[-1,1], s=5, marker='s', facecolors='none', edgecolors='g')
            ax.plot(corners[:,0], corners[:,1], c='y')
            ax.axis('off')
            self.corners_2d_demo_dir[pose_id] = os.path.join(self.temp_dir, f'{uuid.uuid4()}.png')
            fig.savefig(self.corners_2d_demo_dir[pose_id], dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        return img, result_corners, board_size


    def take_pointcloud_image(self, pcd_t, z_rotation:float, width:int=1200, height:int=800, focal_length:float=1200.):
        lidar_cam_intrinsics = np.array([
            [focal_length, 0., width/2],
            [0., focal_length, height/2],
            [0., 0., 1.]
        ])
        lidar_cam_extrinsics = np.identity(4)
        lidar_cam_extrinsics[0:3, 0:3] = Rotation.from_euler('XYZ', [90, 0, 90+z_rotation], degrees=True).as_matrix() # [pitch, roll, yaw]
        lidar_cam_extrinsics[0:3, 3] = [0., 0., 0.]

        rgbd_img = pcd_t.project_to_rgbd_image(
            width=int(lidar_cam_intrinsics[0,2]*2), height=int(lidar_cam_intrinsics[1,2]*2),
            intrinsics=lidar_cam_intrinsics,
            extrinsics=lidar_cam_extrinsics,
            depth_max=10.0, depth_scale=1.0
        )

        intensity_img = np.array(rgbd_img.color, np.float32)[:,:,0]
        depth_img = np.array(rgbd_img.depth, np.float32)
        return intensity_img, depth_img


    def enhance_lidar_image(self, intensity_img:np.ndarray, densify=True, do_redistribute=True, kernel_size:int=3):
        # Redistribute intensity image
        if do_redistribute:
            intensity_img = redistribute_intensity(intensity_img.flatten(), ignore_zeros=True, debug=False).reshape(intensity_img.shape)
        # redistr_intensities = intensity_img[intensity_img!=0].flatten()

        # Densify intensity image
        dense_intensity_img = np.array(intensity_img, np.float32)
        kernel_offset = kernel_size // 2
        if densify and kernel_size > 0:
            for u in range(kernel_offset, dense_intensity_img.shape[0]-kernel_offset):
                for v in range(kernel_offset, dense_intensity_img.shape[1]-kernel_offset):
                    if intensity_img[u, v] != 0:
                        for i in range(u-kernel_offset, u+kernel_offset+1):
                            for j in range(v-kernel_offset, v+kernel_offset+1):
                                if i == u and j == v:
                                    continue
                                if dense_intensity_img[i, j] == 0:
                                    dense_intensity_img[i, j] = dense_intensity_img[u, v]
                                else:
                                    dense_intensity_img[i, j] = (dense_intensity_img[i, j] + dense_intensity_img[u, v]) / 2
            # dense_intensity_img[dense_intensity_img < np.average(redistr_intensities)] = 0.

        return dense_intensity_img


    def detect_lidar_image_chessboard_corners(self, inp_img:np.ndarray, debug=False):
        enhanced_img = self.enhance_lidar_image(inp_img)
        # plt.imsave('lidar_image_corners_test.png', enhanced_img, cmap='gray')
        try:
            _, chessboard_corners, board_size = self.detect_corners(
                enhanced_img,
                min_corner_metric=0.2 if self.use_matlab else 0,
                use_matlab=self.use_matlab,
            )
            # print(chessboard_corners, board_size)
        except Exception as e:
            chessboard_corners = []
            board_size = [0, 0]

        if debug:
            os.makedirs(self.temp_dir, exist_ok=True)
            fig, axs = plt.subplots(2, 3, figsize=(10,8), gridspec_kw={'width_ratios': [4, 1, 1]})
            axs[0][0].set_title('Original intensity image')
            axs[0][0].imshow(inp_img, cmap='gray')
            axs[0][1].boxplot(inp_img[inp_img!=0].flatten(), whis=0.5)
            axs[0][2].hist(inp_img[inp_img!=0].flatten(), bins=100, orientation='horizontal')

            axs[1][0].set_title('Intensity image with distribution correction')
            axs[1][0].imshow(enhanced_img, cmap='gray')
            if len(chessboard_corners) > 0:
                axs[1][0].scatter(chessboard_corners[:,0], chessboard_corners[:,1], s=30, facecolors='none', edgecolors='r')
            axs[1][1].boxplot(enhanced_img[enhanced_img!=0].flatten(), whis=0.5)
            axs[1][2].hist(enhanced_img[enhanced_img!=0].flatten(), bins=100, orientation='horizontal')

            with io.BytesIO() as buf:
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.2)
                plt.close(fig)
                buf.seek(0)
                self.showcase['LiDAR_image'] = f'data:image/png;base64,{base64.b64encode(buf.read()).decode("utf-8")}'
            plt.close(fig)

        return chessboard_corners, board_size


    def automatic_3d_chessboard_detection(self, pcd:o3d.geometry.PointCloud, corners_3D_detector:int=0, debug=False):
        grid_inner_size = self.grid_inner_size
        square_size = self.chessboard_square_size
        
        if corners_3D_detector == 1:
            pcd_down = pcd.voxel_down_sample(voxel_size=0.005)
            pcd_down.estimate_normals()
            grid_inner_size = self.grid_inner_size
            square_size = self.chessboard_square_size
            planar_patches = pcd_down.detect_planar_patches(
                coplanarity_deg = 0.9,
                min_plane_edge_length=grid_inner_size[0]*square_size,
            )
            chessboard_extent = (grid_inner_size+2) * square_size
            for plane_bbox in planar_patches:
                bbox_extent = np.sort(plane_bbox.extent)[::-1]
                if np.abs(bbox_extent[0] - chessboard_extent[0]) <= 0.2 and np.abs(bbox_extent[1] - chessboard_extent[1]) <= 0.2:
                    est_chessboard_bbox = plane_bbox
                    break
        
        elif corners_3D_detector == 2:
            est_chessboard_bbox = self.hybrid_3D_chessboard_detector(pcd, debug=debug)

        elif corners_3D_detector == 0:
            pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd)
            lidar_cam_width = 600
            lidar_cam_height = 360

            chessboard_detected = False
            for z_rotation in np.arange(self.roi_direction-self.roi_angular_width, self.roi_direction+self.roi_angular_width+1, 10.):
                for focal_length in [300, 800]:
                    # Generate LiDAR image
                    intensity_img, depth_img = self.take_pointcloud_image(
                        pcd_t, z_rotation,
                        width=lidar_cam_width, height=lidar_cam_height,
                        focal_length=focal_length
                    )

                    # Detect chessboard corners from LiDAR image
                    start_time = time.time()
                    chessboard_corners, board_size = self.detect_lidar_image_chessboard_corners(
                        intensity_img, debug=True # Keep debug=False
                    )
                    print(f'LiDAR image chessboard corners detection time: {time.time()-start_time:.2f}s')
                    if len(chessboard_corners) <= 2:
                        continue

                    # Verification of detected corners from LiDAR image
                    lidar_cam_extrinsics = tf_matrix(
                        R = Rotation.from_euler('XYZ', [90, 0, 90+z_rotation], degrees=True).as_matrix()
                    )
                    if board_size[0] == grid_inner_size[0]+2 and board_size[1] == grid_inner_size[1]+2:
                        corners_3d_estimates = depth_to_pcd(
                            depth_img, chessboard_corners,
                            fx=focal_length, fy=focal_length,
                            cx=lidar_cam_width/2, cy=lidar_cam_height/2,
                            depth_cam_extrinsics=lidar_cam_extrinsics,
                            remove_depth_outlier=False
                        )
                        est_corners_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(corners_3d_estimates))
                        est_chessboard_bbox = est_corners_pcd.get_oriented_bounding_box(robust=True)
                        est_bbox_extent = est_chessboard_bbox.extent
                        est_bbox_extent = est_bbox_extent[np.argsort(est_bbox_extent)[::-1]]
                        if np.abs(est_bbox_extent[0] - grid_inner_size[0]*square_size) <= 0.2\
                            and np.abs(est_bbox_extent[1] - grid_inner_size[1]*square_size) <= 0.2:
                                # Confirmed that the chessboard is detected entirely
                                est_chessboard_bbox.extent = [
                                    (grid_inner_size[0]+8)*square_size, (grid_inner_size[1]+4)*square_size, 0.3
                                ]
                                chessboard_detected = True

                    elif 0 < board_size[0] <= grid_inner_size[0]+2 and 0 < board_size[1] <= grid_inner_size[1]+2:
                        chessboard_2D_center = np.average(chessboard_corners, axis=0)
                        dist_to_image_center = np.abs(chessboard_2D_center - np.array([lidar_cam_width/2, lidar_cam_height/2], np.float32))
                        if dist_to_image_center[0] > lidar_cam_width * 7./8.:
                            continue
                        partial_3d_corners = depth_to_pcd(
                            depth_img, chessboard_corners,
                            fx=focal_length, fy=focal_length,
                            cx=lidar_cam_width/2, cy=lidar_cam_height/2,
                            depth_cam_extrinsics=lidar_cam_extrinsics,
                            remove_depth_outlier=False
                        )
                        partial_corners_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(partial_3d_corners))
                        partial_bbox = partial_corners_pcd.get_minimal_oriented_bounding_box(robust=True)
                        partial_bbox_extent = partial_bbox.extent
                        extent_order = np.argsort(partial_bbox_extent)[::-1]
                        partial_bbox_extent = partial_bbox_extent[extent_order]
                        if np.abs(partial_bbox_extent[0] - (max(board_size[0], board_size[1])-2)*square_size) <= 0.2\
                            and np.abs(partial_bbox_extent[1] - (min(board_size[0], board_size[1])-2)*square_size) <= 0.2:
                                # Confirmed that the chessboard is detected partially
                                chessboard_detected = True
                                partial_bbox.color = [1,0,0]

                                # Correct the normal orientation of the est bbox
                                world_axes = np.identity(3)
                                bbox_w = partial_bbox.R @ world_axes[extent_order[2]]
                                if np.dot(bbox_w, partial_bbox.center) > 0:
                                    bbox_w = -bbox_w

                                # Correct the short-edge orientation of the est bbox
                                option_0 = partial_bbox.R @ world_axes[extent_order[0]]
                                option_1 = partial_bbox.R @ world_axes[extent_order[1]]
                                angle_0 = np.arccos(np.dot(option_0, [0, 0, 1]))
                                angle_1 = np.arccos(np.dot(option_1, [0, 0, 1]))
                                if angle_0 > np.pi/2: angle_0 = np.pi - angle_0
                                if angle_1 > np.pi/2: angle_1 = np.pi - angle_1
                                if angle_0 == angle_1: # Truly no way else :) :) :)
                                    continue
                                elif angle_0 < angle_1:
                                    bbox_v = option_0
                                else:
                                    bbox_v = option_1
                                if np.dot(bbox_v, [0,0,1]) < 0:
                                    bbox_v = -bbox_v
                                bbox_u = np.cross(bbox_v, bbox_w)
                                est_corners_pcd = o3d.geometry.PointCloud(partial_corners_pcd)

                                # Find better bbox
                                est_chessboard_bbox = o3d.geometry.OrientedBoundingBox()
                                est_chessboard_bbox.center = partial_bbox.center
                                est_chessboard_bbox.R = Rotation.align_vectors([bbox_u, bbox_v], [[1,0,0], [0,1,0]])[0].as_matrix()

                    if chessboard_detected:
                        if debug:
                            self.detect_lidar_image_chessboard_corners(
                                intensity_img, debug=debug
                            )
                        est_chessboard_bbox = bounded_3d_region_growing(
                            pcd,
                            est_chessboard_bbox.center,
                            est_chessboard_bbox.R,
                            [(grid_inner_size[0]+2.5)*square_size, (grid_inner_size[1]+4)*square_size, 0.2],
                            square_size/10.
                        )
                        break

                if chessboard_detected: break
        

        try:
            if est_chessboard_bbox is None:
                return None, None, None, None, None, None
        except:
            # raise ValueError('Could not detect calibration target from input point cloud')
            return None, None, None, None, None, None

        # Filter out points outside of the chessboard plane
        rough_chessboard_pcd = pcd.crop(est_chessboard_bbox)
        _, chessboard_plane_inliers = rough_chessboard_pcd.segment_plane(
            distance_threshold=0.04,
            ransac_n=30,
            num_iterations=500
        )
        chessboard_inliers_pcd = rough_chessboard_pcd.select_by_index(chessboard_plane_inliers)
        chessboard_bbox = chessboard_inliers_pcd.get_minimal_oriented_bounding_box(robust=True)

        # Orientation correction
        world_axes = np.identity(3)
        ax_order = np.argsort(chessboard_bbox.extent)[::-1]
        centroid = np.array(chessboard_bbox.center)

        chessboard_z_axis = chessboard_bbox.R @ world_axes[ax_order[2]]
        chessboard_y_axis = chessboard_bbox.R @ world_axes[ax_order[1]]
        if np.dot(chessboard_z_axis, centroid) > 0:
            chessboard_z_axis = -chessboard_z_axis
        if np.dot(chessboard_y_axis, [0., 0., 1.]) < 0:
            chessboard_y_axis = -chessboard_y_axis
        chessboard_x_axis = np.cross(chessboard_y_axis, chessboard_z_axis)

        # Further crop the outer region
        chessboard_bbox.R = Rotation.align_vectors([chessboard_x_axis, chessboard_y_axis], [[1,0,0], [0,1,0]])[0].as_matrix()
        chessboard_bbox.extent = [
            (grid_inner_size[0]+4)*square_size, (grid_inner_size[1]+8)*square_size, 0.1
        ]
        chessboard_pcd = chessboard_inliers_pcd.crop(chessboard_bbox)
        # rough_chessboard_pcd.paint_uniform_color([0,0,1])
        # chessboard_inliers_pcd.paint_uniform_color([0,1,0])
        chessboard_pcd_intensities = np.array(chessboard_pcd.colors)[:,0]
        chessboard_pcd_intensities = redistribute_intensity(chessboard_pcd_intensities, debug=False)
        chessboard_3d_points = np.array(chessboard_pcd.points)
        # chessboard_bbox.color = np.array([1, 0, 0], np.float64)

        return chessboard_3d_points, chessboard_pcd_intensities, centroid, chessboard_x_axis, chessboard_y_axis, chessboard_z_axis



    def plane_based_chessboard_detection(self, pcd:o3d.geometry.PointCloud, debug=False):
        pcd_down = pcd.voxel_down_sample(voxel_size=0.005)
        pcd_down.estimate_normals()
        grid_inner_size = self.grid_inner_size
        square_size = self.chessboard_square_size
        planar_patches = pcd_down.detect_planar_patches(
            coplanarity_deg = 0.9,
            min_plane_edge_length=grid_inner_size[0]*square_size,
        )
        chessboard_extent = (grid_inner_size+2) * square_size
        print(chessboard_extent)
        for plane_bbox in planar_patches:
            print(plane_bbox.extent)
            bbox_extent = np.sort(plane_bbox.extent)[::-1]
            print(bbox_extent)
            if np.abs(bbox_extent[0] - chessboard_extent[0]) <= 0.2 and np.abs(bbox_extent[1] - chessboard_extent[1]) <= 0.2:
                est_chessboard_bbox = plane_bbox
                break
        
        try:
            est_chessboard_bbox
        except:
            raise ValueError('Could not detect calibration target from input point cloud with plane-based method')
        
        # Filter out points outside of the chessboard plane
        rough_chessboard_pcd = pcd.crop(est_chessboard_bbox)
        _, chessboard_plane_inliers = rough_chessboard_pcd.segment_plane(
            distance_threshold=0.04,
            ransac_n=30,
            num_iterations=500
        )
        chessboard_inliers_pcd = rough_chessboard_pcd.select_by_index(chessboard_plane_inliers)
        chessboard_bbox = chessboard_inliers_pcd.get_minimal_oriented_bounding_box(robust=True)

        # Orientation correction
        world_axes = np.identity(3)
        ax_order = np.argsort(chessboard_bbox.extent)[::-1]
        centroid = np.array(chessboard_bbox.center)

        chessboard_z_axis = chessboard_bbox.R @ world_axes[ax_order[2]]
        chessboard_y_axis = chessboard_bbox.R @ world_axes[ax_order[1]]
        if np.dot(chessboard_z_axis, centroid) > 0:
            chessboard_z_axis = -chessboard_z_axis
        if np.dot(chessboard_y_axis, [0., 0., 1.]) < 0:
            chessboard_y_axis = -chessboard_y_axis
        chessboard_x_axis = np.cross(chessboard_y_axis, chessboard_z_axis)

        # Further crop the outer region
        chessboard_bbox.R = Rotation.align_vectors([chessboard_x_axis, chessboard_y_axis], [[1,0,0], [0,1,0]])[0].as_matrix()
        chessboard_bbox.extent = [
            (grid_inner_size[0]+2)*square_size, (grid_inner_size[1]+2)*square_size, 0.1
        ]
        chessboard_pcd = chessboard_inliers_pcd.crop(chessboard_bbox)
        chessboard_pcd_intensities = np.array(chessboard_pcd.colors)[:,0]
        chessboard_pcd_intensities = redistribute_intensity(chessboard_pcd_intensities, debug=False)
        chessboard_3d_points = np.array(chessboard_pcd.points)

        return chessboard_3d_points, chessboard_pcd_intensities, centroid, chessboard_x_axis, chessboard_y_axis, chessboard_z_axis



    def chessboard_cost_function(self,
        input_x:np.ndarray,
        orig_chessboard_uv:np.ndarray,
        orig_chessboard_intensities:np.ndarray,
        # centroid:np.ndarray,
        # x_axis:np.ndarray, y_axis:np.ndarray, z_axis:np.ndarray,
        debug=False, pose_id:int=None
    ):
        grid_inner_size = self.grid_inner_size
        square_size = self.chessboard_square_size
        num_squares = (grid_inner_size[0]+2) * (grid_inner_size[1]+2)
        # orig_chessboard_uv = get_chessboard_2d_projection(
        #     pcd, centroid, x_axis, y_axis, z_axis,
        #     input_x,
        #     grid_inner_size, square_size
        # )

        # Transform the projected chessboard pcd
        transformed_chessboard_uv = orig_chessboard_uv.copy()
        rot_mat = np.array([
            [np.cos(input_x[0]), -np.sin(input_x[0])],
            [np.sin(input_x[0]), np.cos(input_x[0])]
        ])
        transformed_chessboard_uv = (rot_mat @ transformed_chessboard_uv.T).T
        transformed_chessboard_uv -= input_x[1:3]

        # Process the transformed projected chessboard pcd in 2D
        orig_chessboard_square_ids = get_chessboard_square_id(transformed_chessboard_uv, grid_inner_size, square_size)
        valid_square_ids = np.where(orig_chessboard_square_ids != -1)[0]
        N = len(valid_square_ids)
        chessboard_uv = transformed_chessboard_uv[valid_square_ids]
        chessboard_square_ids = orig_chessboard_square_ids[valid_square_ids]
        chessboard_intensities = orig_chessboard_intensities[valid_square_ids]

        squares_uv = get_square_uv(grid_inner_size, square_size)
        chessboard_square_uv = np.zeros_like(chessboard_uv)
        for i in range(len(chessboard_square_uv)):
            chessboard_square_uv[i] = squares_uv[chessboard_square_ids[i]]

        squares_colours = determine_squares_colours(chessboard_intensities, chessboard_square_ids, grid_inner_size)

        chessboard_square_intensities = np.zeros_like(chessboard_intensities)
        for i in range(len(chessboard_square_intensities)):
            chessboard_square_intensities[i] = squares_colours[chessboard_square_ids[i]]

        dist_sq = (chessboard_uv[:,0] - chessboard_square_uv[:,0])**2 + (chessboard_uv[:,1] - chessboard_square_uv[:,1])**2
        dist = np.sqrt(dist_sq)
        I_coef = 10.0
        cost_arr = (1.0 + dist/square_size)**2 + I_coef*(1.0 + chessboard_intensities - chessboard_square_intensities)**2

        def find_cost_per_square(cost_arr:np.ndarray):
            result = 0.
            for square_id in range(num_squares):
                points_ids_in_square = np.where(chessboard_square_ids==square_id)[0]
                if len(points_ids_in_square) > 0:
                    result += np.sum(cost_arr[points_ids_in_square])**2 / len(points_ids_in_square)
                else:
                    result += 0.
            return result

        cost = find_cost_per_square(cost_arr) / N

        if debug and pose_id is not None:
            print(cost)
            fig, ax = plt.subplots(figsize=(6,4))
            axs = [ax]
            axs[0].set_facecolor(np.array([70, 130, 180], float)/255.)
            axs[0].scatter(chessboard_uv[:,0], chessboard_uv[:,1], c=chessboard_intensities, s=10, cmap='gray', vmin=0, vmax=1)
            axs[0].axis('equal')
            axs[0].grid(color='y')
            axs[0].set_title('cost = {:.3f}'.format(cost))
            axs[0].set_xlabel('u [m]')
            axs[0].set_ylabel('v [m]')
            axs[0].set_xlim(np.array([-grid_inner_size[0]/2-1, grid_inner_size[0]/2+1])*square_size)
            axs[0].set_ylim(np.array([-grid_inner_size[1]/2-1, grid_inner_size[1]/2+1])*square_size)
            axs[0].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(square_size))
            with io.BytesIO() as buf:
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.2)
                plt.close(fig)
                buf.seek(0)
                img_base64 = f'data:image/png;base64,{base64.b64encode(buf.getbuffer()).decode()}'
                if pose_id == 'before_opt':
                    self.showcase['before_opt'] = img_base64
                elif pose_id == 'after_opt':
                    self.showcase['after_opt'] = img_base64
                else:
                    self.opt_result_vis[pose_id] = img_base64
        return cost


    def optimise_chessboard_tf(self,
        pose_id:int, pcd:np.ndarray,
        orig_chessboard_intensities:np.ndarray, centroid:np.ndarray,
        x_axis:np.ndarray, y_axis:np.ndarray, z_axis:np.ndarray,
        upper_bounds:np.ndarray = None, debug=False
    ):
        grid_inner_size = self.grid_inner_size
        square_size = self.chessboard_square_size
        if not isinstance(upper_bounds, np.ndarray):
            upper_bounds = np.array([20./180.*np.pi, 0.8*square_size, 0.8*square_size])

        # Project 3D chessboard pcd to 2D
        orig_chessboard_uv = get_chessboard_2d_projection(
            pcd, centroid, x_axis, y_axis, z_axis,
            np.zeros(3),
            grid_inner_size, square_size
        )

        # optimiser_input = {
        #     'pcd': pcd.tolist(),
        #     'orig_chessboard_intensities': orig_chessboard_intensities.tolist(),
        #     'centroid': centroid.tolist(),
        #     'x_axis': x_axis.tolist(),
        #     'y_axis': y_axis.tolist(),
        #     'z_axis': z_axis.tolist(),
        # }
        # with open('optimiser_input.json', 'w') as f:
        #     json.dump(optimiser_input, f, indent=4)
        with mp.Pool(8) as mp_pool:
            start_time = time.time()
            result = scipy.optimize.differential_evolution(
                func = chessboard_cost_function,
                bounds = scipy.optimize.Bounds(-upper_bounds, upper_bounds),
                args = (
                    orig_chessboard_uv, orig_chessboard_intensities,
                    grid_inner_size, square_size,
                ),
                workers = mp_pool.map,
            )
        print(f'Optimisation time: {time.time()-start_time:.2f}s')

        # os.makedirs(self.temp_dir, exist_ok=True)
        if debug:
            print(result)
            self.showcase['before_opt'] = os.path.join(self.temp_dir, f'{uuid.uuid4()}.png')
            self.chessboard_cost_function(
                np.zeros(3),
                orig_chessboard_uv, orig_chessboard_intensities,
                debug=True, pose_id='before_opt'
            )

        # opt_save_dir = os.path.join(self.temp_dir, f'corners_3D_{pose_id}.png')
        if debug:
            # self.showcase['after_opt'] = opt_save_dir
            opt_pose_id = 'after_opt'
        else:
            opt_pose_id = pose_id
        # else:
        #     self.opt_result_vis[pose_id] = opt_save_dir
        opt_cost = self.chessboard_cost_function(
            result.x,
            orig_chessboard_uv, orig_chessboard_intensities,
            debug=True, pose_id=opt_pose_id
        )
        self.corners_3D_costs[pose_id] = opt_cost

        # Convert optimised tf from rad to deg
        opt_tf = result.x.copy()
        opt_tf[0] *= 180./np.pi
        # print('Optimiser output:', opt_tf, 'cost:', opt_cost)
        return opt_tf, opt_cost




    def hybrid_3D_chessboard_detector(
            self,
            pcd:o3d.geometry.PointCloud,
            debug=False,
    ):
        pcd_down = pcd.voxel_down_sample(voxel_size=0.005)
        pcd_down.estimate_normals()
        grid_inner_size = self.grid_inner_size
        square_size = self.chessboard_square_size
        planar_patches = pcd_down.detect_planar_patches(
            coplanarity_deg = 0.9,
            min_plane_edge_length=grid_inner_size[0]*square_size,
        )
        print(f'Found {len(planar_patches)} planar patches')
        chessboard_extent = (grid_inner_size+2) * square_size
        expanded_chessboard_extent = np.array(
            [(grid_inner_size[0]+4) * square_size, (grid_inner_size[1]+8) * square_size, 0.1]
        )
        world_axes = np.identity(3)

        for plane_bbox in planar_patches:
            centroid = np.array(plane_bbox.center)
            ax_order = np.argsort(plane_bbox.extent)[::-1]
            bbox_extent = np.array(plane_bbox.extent)[ax_order]
            if np.abs(bbox_extent[0] - chessboard_extent[0]) > 0.2 or np.abs(bbox_extent[1] - chessboard_extent[1]) > 0.2:
                continue
            z_axis = plane_bbox.R @ world_axes[ax_order[2]]
            y_axis = plane_bbox.R @ world_axes[ax_order[1]]
            if np.dot(z_axis, centroid) > 0:
                z_axis = -z_axis
            if np.dot(y_axis, [0., 0., 1.]) < 0:
                y_axis = -y_axis
            x_axis = np.cross(y_axis, z_axis)

            new_bbox = o3d.geometry.OrientedBoundingBox(
                center = centroid,
                R = plane_bbox.R,
                extent = expanded_chessboard_extent[ax_order]
            )
            
            chessboard_pcd = pcd.crop(new_bbox)
            chessboard_intensities = np.array(chessboard_pcd.colors)[:,0]
            chessboard_intensities = redistribute_intensity(chessboard_intensities, debug=False)

            chessboard_uv = get_chessboard_2d_projection(
                np.array(chessboard_pcd.points), centroid,
                x_axis, y_axis, z_axis
            )
            fig, ax = plt.subplots(figsize=(3,2))
            axs = [ax]
            axs[0].scatter(chessboard_uv[:,0], chessboard_uv[:,1], c=chessboard_intensities, s=10, cmap='gray', vmin=0, vmax=1)
            axs[0].axis('equal')
            axs[0].axis('off')
            
            with io.BytesIO() as buf:
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2, facecolor='black')
                plt.close(fig)
                buf.seek(0)
                img_arr = plt.imread(buf) / 255.
                img_arr = img_arr[:,:,0].astype(np.float32)
                img_arr = self.enhance_lidar_image(img_arr, densify=True, do_redistribute=False, kernel_size=2)

            # plt.imsave('test.png', img_arr, cmap='gray')
            _, corners, board_size = self.detect_corners(
                img_arr,
                use_matlab=self.use_matlab,
                min_corner_metric=0.2,
                use_gpu=False
            )
            if len(corners) > 0:
                est_chessboard_bbox = new_bbox
                break

        if debug:
            self.showcase['LiDAR_image'] = os.path.join(self.temp_dir, f'{uuid.uuid4()}.png')
            plt.imsave(self.showcase['LiDAR_image'], img_arr, cmap='gray')

        try:
            return est_chessboard_bbox
        except:
            return None




    def find_and_optimise_3D_corners(
            self,
            pose_id:int,
            pcd:o3d.geometry.PointCloud,
            corners_3D_detector:int=0,
            debug=False
    ):
        grid_inner_size = self.grid_inner_size
        square_size = self.chessboard_square_size

        try:
            start_time = time.time()
            chessboard_3d_points, chessboard_intensities, chessboard_centroid, \
            chessboard_x_axis, chessboard_y_axis, chessboard_z_axis = self.automatic_3d_chessboard_detection(
                pcd, corners_3D_detector=corners_3D_detector, debug=debug
            )
            print(f'3D chessboard corners detection time: {time.time()-start_time:.2f}s')
        except:
            return [], []
        if chessboard_x_axis is None or chessboard_y_axis is None or chessboard_z_axis is None:
            return [], []

        try:
            optimised_chessboard_tf, opt_cost = self.optimise_chessboard_tf(
                pose_id, chessboard_3d_points, chessboard_intensities, chessboard_centroid,
                chessboard_x_axis, chessboard_y_axis, chessboard_z_axis,
                debug=debug
            )
        except Exception as e:
            print(e)
            raise e
            return [], []
        chessboard_pose = get_chessboard_pose(
            optimised_chessboard_tf, chessboard_centroid, chessboard_x_axis, chessboard_y_axis, chessboard_z_axis,
        )
        chessboard_corners_3d = get_chessboard_corners_3d(
            chessboard_pose, grid_inner_size, square_size, debug=False
        )
        return chessboard_pose, chessboard_corners_3d



    async def detect_3d_corners(self,
            LiDAR_name:str,
            selected_poses:'list[int]'=[],
            skipped_poses:'list[int]'=[],
            corners_3D_detector:int=0,
            demo=False,
            queue:asyncio.Queue=None
    ):
        if len(selected_poses) == 0:
            selected_poses = [i for i in range(self.num_poses)]

        if not demo: self.LiDAR_name = LiDAR_name
        pcds_o3d = self.point_clouds_o3d[LiDAR_name]
        if len(pcds_o3d) == 0:
            print('Converting point clouds to open3d format')
            for i in range(self.num_poses):
                I = self.data.point_clouds[LiDAR_name][i][:,3]
                I /= np.max(I)
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(self.data.point_clouds[LiDAR_name][i][:,:3])
                temp_pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap('gray')(I)[:, 0:3])
                self.point_clouds_o3d[LiDAR_name].append(temp_pcd)
            pcds_o3d = self.point_clouds_o3d[LiDAR_name]

        async def single_3d_corners_detector(pose_id, pcd_o3d:o3d.geometry.PointCloud):
            await asyncio.sleep(0.001)
            try:
                temp_chessboard_pose, temp_chessboard_3d_corners = self.find_and_optimise_3D_corners(
                    pose_id, pcd_o3d, corners_3D_detector, debug=demo
                )
            except Exception as e:
                raise e
                return None, None
            return temp_chessboard_pose, temp_chessboard_3d_corners

        for count, i in enumerate(selected_poses):
            if i not in skipped_poses:
                print('Processing point cloud', i)
                task = asyncio.create_task(single_3d_corners_detector(i, pcds_o3d[i]))
                chessboard_pose, chessboard_3d_corners = await task
                if isinstance(chessboard_pose, np.ndarray) and isinstance(chessboard_3d_corners, np.ndarray):
                    self.all_chessboard_poses[i] = chessboard_pose
                    self.all_chessboard_3d_corners[i] = chessboard_3d_corners
                    if i not in self.valid_LiDAR_poses:
                        self.valid_LiDAR_poses.append(i)
                else:
                    self.all_chessboard_poses[i] = None
                    self.all_chessboard_3d_corners[i] = None
                    print('Skipping point cloud', i)
            if queue is not None:
                await queue.put((count+1)/len(selected_poses))

        await asyncio.sleep(0.001)
    


    def export_pcd_chessboard_labels(self, selected_poses):
        if len(selected_poses) == 0:
            raise ValueError('No selected poses')
        chessboard_bbox_extent = [
            (self.grid_inner_size[0]+2.5)*self.chessboard_square_size,
            (self.grid_inner_size[1]+2.5)*self.chessboard_square_size,
            0.07
        ]
        fields = ("x", "y", "z", "intensity", "label")
        types = (np.float32, np.float32, np.float32, np.float32, np.uint8)

        datadir_folder_name = os.path.basename(self.data.data_dir)
        save_dir = os.path.join('/home/quang/chessboard_det_train', datadir_folder_name)
        try:
            shutil.rmtree(save_dir)
        except: pass
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'velodyne'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'pcds'), exist_ok=True)

        for pose_id in selected_poses:
            chessboard_pose = self.all_chessboard_poses[pose_id]
            if chessboard_pose is None:
                raise ValueError
            chessboard_bbox = o3d.geometry.OrientedBoundingBox(
                center = chessboard_pose[:3,3],
                R = chessboard_pose[:3,:3],
                extent = chessboard_bbox_extent
            )
            chessboard_points_ids:'list[int]' = chessboard_bbox.get_point_indices_within_bounding_box(
                                                self.point_clouds_o3d[self.LiDAR_name][pose_id].points)
            pcd_arr = self.data.point_clouds[self.LiDAR_name][pose_id].astype(np.float32)
            chessboard_labels = np.zeros(pcd_arr.shape[0], np.uint32)
            chessboard_labels[chessboard_points_ids] = 1

            pcd_arr.tofile(os.path.join(save_dir, 'velodyne', f'{self.LiDAR_name}_{pose_id}.bin'))
            chessboard_labels.tofile(os.path.join(save_dir, 'labels', f'{self.LiDAR_name}_{pose_id}.label'))
            
            pcd_with_labels = np.append(pcd_arr, chessboard_labels.reshape((-1,1)).astype(np.uint8), axis=1)
            # print(pcd_with_labels.shape)
            pcd = pypcd4.PointCloud.from_points(pcd_with_labels, fields, types)
            # print(pcd.fields, pcd.types)
            pcd.save(os.path.join(save_dir, 'pcds', f'{self.LiDAR_name}_{pose_id}.pcd'))




    # 2D corners detection
    async def detect_2d_corners(self,
            selected_poses:'list[int]'=[],
            skipped_poses:'list[int]'=[],
            queue:asyncio.Queue=None
    ):
        if len(selected_poses) == 0:
            selected_poses = self.valid_LiDAR_poses
        downscale = 1

        async def single_2d_corners_detector(pose_id:int, img:np.ndarray):
            await asyncio.sleep(0.001)
            try:
                if downscale != 1:
                    temp_img = np.array(cv2.resize(img, (img.shape[1]//downscale, img.shape[0]//downscale)))
                else:
                    temp_img = img
                temp_undst_img, temp_undistorted_2D_corners, temp_board_size = self.detect_corners(
                    temp_img,
                    min_corner_metric = 0.1 if self.use_matlab else 0.0001,
                    use_matlab=self.use_matlab,
                    undistort=True,
                    refine=True,
                    demo=True,
                    pose_id=pose_id
                )
                print(f'Processed pose {pose_id}')
            except Exception as e:
                return None, None, None
            return temp_undst_img, temp_undistorted_2D_corners, temp_board_size

        self.valid_cam_poses = []
        for count, i in enumerate(selected_poses):
            self.undistorted_images[i] = None
            self.all_chessboard_2d_corners[i] = None
            if i not in skipped_poses:
                task = asyncio.create_task(single_2d_corners_detector(i, self.data.images[i],))
                undst_img, undistorted_2D_corners, board_size = await task
                if isinstance(undistorted_2D_corners, np.ndarray) and isinstance(board_size, np.ndarray):
                    # print(board_size, self.grid_inner_size)
                    if board_size[0] == self.grid_inner_size[0]+2 and board_size[1] == self.grid_inner_size[1]+2:
                    # if board_size[0]*board_size[1] == (self.grid_inner_size[0]+2)*(self.grid_inner_size[1]+2):
                        if i not in self.valid_cam_poses:
                            self.valid_cam_poses.append(i)
                        self.undistorted_images[i] = undst_img
                        self.all_chessboard_2d_corners[i] = undistorted_2D_corners
                        # print(f'Pose {i} 2D corners validated')
            if queue is not None:
                await queue.put((count+1)/len(selected_poses))



    async def find_camera_pose_PnP(self, selected_poses:'list[int]'=[], reversed_corners_poses:'list[int]'=[]):
        # Perform PnP on detected 3D corners and undistorted 2D corners
        assert len(selected_poses) > 0
        print('Estimating initial camera pose...')
        
        for i in reversed_corners_poses:
            if i not in selected_poses: continue
            self.all_chessboard_2d_corners[i] = self.all_chessboard_2d_corners[i][::-1]

        corners_3D = np.vstack([self.all_chessboard_3d_corners[i] for i in selected_poses])
        corners_2D = np.vstack([self.all_chessboard_2d_corners[i] for i in selected_poses])

        if False: # Estimate initial cam pose with MATLAB
            import matlab
            cam_intrinsics = self.matlab_engine.cameraIntrinsics(
                # [fx, fy]
                matlab.double([self.cam_intrinsics.FocalLength[0], self.cam_intrinsics.FocalLength[1]]),
                # [cx, cy]
                matlab.double([self.cam_intrinsics.PrinciplePoint[0], self.cam_intrinsics.PrinciplePoint[1]]),
                matlab.double(self.cam_intrinsics.ImageSize[::-1].tolist()),
                'RadialDistortion', matlab.double(self.cam_intrinsics.RadialDistortion.tolist()),
                'TangentialDistortion', matlab.double(self.cam_intrinsics.TangentialDistortion.tolist()),
                'Skew', self.cam_intrinsics.Skew,
            )
            world_corners_ml = matlab.double(corners_3D.tolist())
            image_corners_ml = matlab.double(corners_2D.tolist())
            cam_pose_raw, inlierIdx = self.matlab_engine.estworldpose(
                image_corners_ml, world_corners_ml, cam_intrinsics,
                'MaxReprojectionError', 20.0,
                'MaxNumTrials', 5000,
                nargout = 2
            )
            inlierIdx = np.array(inlierIdx)
            print('Num inliers:', inlierIdx.shape[0])
            init_cam_pose = np.array(self.matlab_engine.getfield(cam_pose_raw, 'A'))
            inv_init_cam_pose = np.linalg.inv(np.array(init_cam_pose))
            init_rvec = Rotation.from_matrix(inv_init_cam_pose[:3,:3]).as_rotvec()
            init_tvec = inv_init_cam_pose[:3,3]

        if True: # Use OpenCV PnP
            ret_val, init_rvec, init_tvec = cv2.solvePnP(
                # corners_3D_p3p, corners_2D_p3p,
                objectPoints = corners_3D,
                imagePoints = corners_2D,
                cameraMatrix = self.cam_intrinsics.IntrinsicMatrix,
                distCoeffs = self.cam_intrinsics.DistCoeffs,
                flags = cv2.SOLVEPNP_SQPNP,
            )
            if not ret_val:
                raise ValueError('Could not estimate initial camera pose')

        if False:
            # Use OpenGV PnP
            inv_K = np.linalg.inv(self.cam_intrinsics.IntrinsicMatrix)
            tmp_corners_2D = np.hstack([corners_2D, np.ones((corners_2D.shape[0], 1))])
            img_bearing_vectors = np.dot(inv_K, tmp_corners_2D.T).T
            img_bearing_vectors /= np.linalg.norm(img_bearing_vectors, axis=1, keepdims=True)

            pcd_bearing_vectors = corners_3D / np.linalg.norm(corners_3D, axis=1, keepdims=True)
            # UPnP_result = pyopengv.absolute_pose_ransac(
            #     img_bearing_vectors, pcd_bearing_vectors, 'EPNP',
            #     1.0 - np.cos(np.arctan(np.sqrt(2.0)*0.5 / np.average(self.cam_intrinsics.FocalLength))),
            #     100000
            # )
            UPnP_result = pyopengv.absolute_pose_upnp(img_bearing_vectors, pcd_bearing_vectors)
            tf = np.vstack((UPnP_result[0], np.array([0,0,0,1])))
            inv_tf = np.linalg.inv(tf)
            init_rvec = Rotation.from_matrix(inv_tf[:3,:3]).as_rotvec()
            init_tvec = inv_tf[:3,3]


        print(init_rvec); print(init_tvec)
        print('First estimation reprj error:', self.get_PnP_reprojection_error(
            corners_3D, corners_2D, init_rvec, init_tvec
        ))
        if False:
            lm_critera = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)
            rvec, tvec = cv2.solvePnPRefineLM(
                corners_3D, corners_2D,
                self.cam_intrinsics.IntrinsicMatrix,
                self.cam_intrinsics.DistCoeffs,
                init_rvec, init_tvec,
                criteria=lm_critera
            )
        else:
            rvec, tvec = self.optimise_camera_pose(
                corners_3D, corners_2D, init_rvec, init_tvec
            )
        print('Optimised rvec:', rvec)
        print('Optimised tvec:', tvec)
        cam_pose = np.identity(4)
        cam_pose[:3,:3] = Rotation.from_rotvec(rvec.flatten()).as_matrix()
        cam_pose[:3,3] = tvec.flatten()
        cam_pose = np.linalg.inv(cam_pose)
        print(cam_pose)
        await asyncio.sleep(0.001)
        return cam_pose, rvec, tvec



    def optimise_camera_pose(self, corners_3D:np.ndarray, corners_2D:np.ndarray, init_rvec:np.ndarray, init_tvec:np.ndarray):
        print('Optimising camera pose...')
        init_rvec = np.array(init_rvec, np.float32).flatten()
        init_tvec = np.array(init_tvec, np.float32).flatten()
        corners_3D = corners_3D.reshape((-1, 3))
        corners_2D = corners_2D.reshape((-1, 2))
        assert corners_3D.shape[0] == corners_2D.shape[0]
        N = corners_3D.shape[0]
        N_sq = N**2
        orig_intrinsic_matrix = self.cam_intrinsics.IntrinsicMatrix.copy()

        def reprojection_error(x):
            rvec = init_rvec + x[0:3]
            tvec = init_tvec + x[3:6]
            # intrinsic_matrix = np.array(orig_intrinsic_matrix, np.float32)
            # intrinsic_matrix[0,0] *= (1 + x[6])
            # intrinsic_matrix[1,1] *= (1 + x[6])
            projected_corners, _ = cv2.projectPoints(
                corners_3D,
                rvec, tvec,
                self.cam_intrinsics.IntrinsicMatrix,
                self.cam_intrinsics.DistCoeffs
            )
            projected_corners = np.array(projected_corners).reshape((-1,2))
            return np.sum(np.sqrt(np.sum((corners_2D - projected_corners)**2, axis=1))) / N

        upper_bounds = np.array([np.pi/3, np.pi/3, np.pi/3, 1, 1, 1], dtype=np.float32)
        result = scipy.optimize.differential_evolution(
            func = reprojection_error,
            x0 = np.zeros(6),
            bounds = scipy.optimize.Bounds(-upper_bounds, upper_bounds),
            tol = 1e-3,
        )
        print(result)
        opt_rvec = init_rvec + result.x[0:3]
        opt_tvec = init_tvec + result.x[3:6]
        # self.cam_intrinsics.IntrinsicMatrix[0,0] *= (1 + result.x[6])
        # self.cam_intrinsics.IntrinsicMatrix[1,1] *= (1 + result.x[6])
        # self.cam_intrinsics.FocalLength *= (1 + result.x[6])
        return opt_rvec, opt_tvec




    def get_PnP_reprojection_error(self,
        corners_3D:np.ndarray, corners_2D:np.ndarray,
        rvec:np.ndarray=None, tvec:np.ndarray=None,
        cam_pose:np.ndarray=None,
        demo=False, demo_img:np.ndarray=None, demo_save_dir:str=None
    ):
        corners_3D = corners_3D.reshape((-1, 3))
        corners_2D = corners_2D.reshape((-1, 2))
        assert corners_3D.shape[0] == corners_2D.shape[0]
        N = corners_3D.shape[0]
        if rvec is None or tvec is None:
            assert cam_pose is not None
            inv_cam_pose = np.linalg.inv(cam_pose)
            rvec = Rotation.from_matrix(inv_cam_pose[:3,:3]).as_rotvec()
            tvec = inv_cam_pose[:3,3]
        else:
            if not isinstance(rvec, np.ndarray):
                rvec = np.array(rvec, np.float32).flatten()
            if not isinstance(tvec, np.ndarray):
                tvec = np.array(tvec, np.float32).flatten()
        projected_corners, _ = cv2.projectPoints(
            corners_3D,
            rvec, tvec,
            self.cam_intrinsics.IntrinsicMatrix,
            self.cam_intrinsics.DistCoeffs
        )
        projected_corners = np.array(projected_corners).reshape(-1,2)
        reprojection_error = np.sum(np.sqrt(np.sum((corners_2D - projected_corners)**2, axis=1))) / N

        if demo and demo_img is not None and demo_save_dir is not None:
            # demo_img = self.cam_intrinsics.undistort_image(demo_img)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.axis('off')
            if demo_img.ndim == 2 or demo_img.shape[2] == 1:
                ax.imshow(demo_img, cmap='gray')
            else:
                ax.imshow(demo_img)
            ax.scatter(
                corners_2D[:,0], corners_2D[:,1], s=20, facecolors='none', edgecolors='g',
                label='Detected corners in 2D image'
            )
            ax.scatter(
                projected_corners[:,0], projected_corners[:,1], s=5, marker='x', c='r',
                label='Projected corners from 3D point cloud'
            )
            ax.legend(title=f'Re-projection error = {reprojection_error:.3f}')
            fig.savefig(demo_save_dir, dpi=200, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        return reprojection_error



    async def calibrate_LiDAR_camera_extrinsics(self,
            selected_poses:'list[int]'=[],
            reversed_corners_poses:'list[int]'=[],
            queue:asyncio.Queue=None
    ):
        if not isinstance(selected_poses, list) or len(selected_poses) == 0:
            selected_poses = self.valid_cam_poses
        else:
            selected_poses = [i for i in selected_poses if i in self.valid_cam_poses]
        self.opt_cam_pose, self.opt_rvec, self.opt_tvec = await self.find_camera_pose_PnP(
            selected_poses, reversed_corners_poses
        )
        if queue is not None:
            await queue.put(0.5)

        self.PnP_reprojection_errors = [None for _ in range(self.num_poses)]
        self.PnP_reprojection_images = [None for _ in range(self.num_poses)]
        self.fusion_images = [None for _ in range(self.num_poses)]

        for i in selected_poses:
            self.PnP_reprojection_images[i] = os.path.join(self.temp_dir, f'{uuid.uuid4()}.png')
            self.PnP_reprojection_errors[i] = self.get_PnP_reprojection_error(
                self.all_chessboard_3d_corners[i], self.all_chessboard_2d_corners[i],
                self.opt_rvec, self.opt_tvec,
                demo=True,
                demo_img=self.data.images[i],
                demo_save_dir=self.PnP_reprojection_images[i]
            )
            self.fusion_images[i] = os.path.join(self.temp_dir, f'{uuid.uuid4()}.png')
            self.fusion_images[i] = self.project_point_cloud_onto_image(
                self.data.point_clouds[self.LiDAR_name][i],
                self.data.images[i],
                self.opt_cam_pose,
                depth_max=15.
            )
            if queue is not None:
                await queue.put((i+1)/len(selected_poses) / 2 + 0.5)

        self.save_calibration_results()
        await asyncio.sleep(0.001)



    def save_calibration_results(self):
        # Save calibration results
        self.calibration_results = {
            'Camera Name': self.data.cam_name,
            'Intrinsic Matrix': self.cam_intrinsics.IntrinsicMatrix.tolist(),
            'Distortion Coefficients': self.cam_intrinsics.DistCoeffs.tolist(),
            'Image Size': self.cam_intrinsics.ImageSize.tolist(),
            'LiDAR Name': self.LiDAR_name,
            'Camera Pose': self.opt_cam_pose.tolist(),
            'Extrinsic rvec': self.opt_rvec.tolist(),
            'Extrinsic tvec': self.opt_tvec.tolist(),
            'Extrinsic quaternion': Rotation.from_rotvec(self.opt_rvec.flatten()).as_quat().tolist(),
        }
        result_dir = os.path.join(self.data.data_dir, 'calib', self.data.cam_name, self.LiDAR_name, 'extrinsics.json')
        with open(result_dir, 'w') as f:
            json.dump(self.calibration_results, f, indent=4)
        print(f'Calibration results saved to {result_dir}')




    def project_point_cloud_onto_image(
            self, pcd:np.ndarray, img:np.ndarray,
            cam_pose:np.ndarray,
            depth_max=35., point_size=0.05,
            fusion_attribute:str='depth'
    ):
        if not self.raw_images_already_undistorted:
            img = self.cam_intrinsics.undistort_image(img)
        device = o3d.core.Device("CPU:0")
        pcd_t = o3d.t.geometry.PointCloud(device)
        pcd_t.point.positions = o3d.core.Tensor(pcd[:,:3], o3d.core.float32, device)
        if fusion_attribute == 'depth':
            colours = np.zeros((pcd.shape[0], 3))
        elif fusion_attribute == 'intensity':
            I = pcd[:,3]
            I /= np.max(I)
            colours = np.vstack((I,I,I)).T
        elif fusion_attribute == 'z-coordinates':
            z = pcd[:,2]
            z -= np.min(z)
            z /= np.max(z)
            colours = np.vstack((z,z,z)).T
        pcd_t.point.colors = o3d.core.Tensor(colours, o3d.core.float32, device)

        rgbd_img = pcd_t.project_to_rgbd_image(
            width = self.cam_intrinsics.ImageSize[0],
            height = self.cam_intrinsics.ImageSize[1],
            intrinsics = self.cam_intrinsics.IntrinsicMatrix,
            extrinsics = np.linalg.inv(cam_pose),
            depth_max=depth_max, depth_scale=1.0
        )
        depth_img = np.array(rgbd_img.depth, np.float32)

        valid_pixels = np.where(depth_img != 0)

        fig, ax = plt.subplots(figsize=(10,10))
        ax.axis('off')
        if img.ndim == 2 or img.shape[2] == 1:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        if fusion_attribute == 'depth':
            ax.scatter(valid_pixels[1], valid_pixels[0], s=point_size, c=depth_img[valid_pixels], cmap='nipy_spectral')
        elif fusion_attribute == 'intensity':
            LiDAR_intensity_img = np.array(rgbd_img.color, np.float32)
            if LiDAR_intensity_img.ndim == 3:
                LiDAR_intensity_img = LiDAR_intensity_img[:,:,0]
            LiDAR_intensity_img = self.enhance_lidar_image(LiDAR_intensity_img, densify=False)
            # print(LiDAR_intensity_img.shape)
            ax.scatter(
                valid_pixels[1], valid_pixels[0],
                s=point_size, c=LiDAR_intensity_img[valid_pixels[0], valid_pixels[1]],
                cmap='nipy_spectral'
            )
        elif fusion_attribute == 'z-coordinates':
            height_img = np.array(rgbd_img.color, np.float32)
            if height_img.ndim == 3:
                height_img = height_img[:,:,0]
            ax.scatter(
                valid_pixels[1], valid_pixels[0],
                s=point_size, c=height_img[valid_pixels[0], valid_pixels[1]],
                cmap='nipy_spectral'
            )
        else:
            raise NotImplementedError

        result_dir = os.path.join(self.temp_dir, f'{uuid.uuid4()}.png')
        fig.savefig(result_dir, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return result_dir




    def colour_point_cloud(self, pcd:np.ndarray, image:np.ndarray, cam_pose:np.ndarray, retina_correction=False):
        ''' Add colours (RGB/thermal) from image to point cloud '''
        if retina_correction:
            raise NotImplementedError
        lidar_to_cam = np.linalg.inv(cam_pose)
        pcd_points = np.hstack([pcd[:,:3], np.ones((pcd.shape[0],1))])
        cam_3d_points = (lidar_to_cam @ pcd_points.T).T
        cam_2d_points = (self.cam_intrinsics.IntrinsicMatrix @ cam_3d_points[:,:3].T).T
        valid_w = np.array(cam_2d_points[:,2] >= 0)
        cam_2d_points /= cam_2d_points[:,2].reshape(-1,1)
        cam_2d_points = cam_2d_points[:,0:2]

        if self.raw_images_already_undistorted:
            cam_2d_points = self.cam_intrinsics.undistort_points(cam_2d_points)
        cam_2d_points = np.round(cam_2d_points).astype(int)

        image_size = self.cam_intrinsics.ImageSize
        valid_u = np.logical_and(0 <= cam_2d_points[:,0], cam_2d_points[:,0] < image_size[0])
        valid_v = np.logical_and(0 <= cam_2d_points[:,1], cam_2d_points[:,1] < image_size[1])
        valid_uv = np.logical_and(valid_u, valid_v)
        valid_2d_points = np.where(np.logical_and(valid_uv, valid_w) == True)[0]

        colours = np.zeros((pcd.shape[0], 3))
        if image.ndim==2 or image.shape[2]==1:  # Thermal image
            pixel_colours = np.zeros((image.shape[0], image.shape[1], 3))
            for i in range(3):
                pixel_colours[:,:,i] = image/255.0
        elif len(image.shape) == 3:
            pixel_colours = image/255.0

        valid_cam_2d_points = cam_2d_points[valid_2d_points]
        # if self.raw_images_already_undistorted:
        #     valid_cam_2d_points = self.cam_intrinsics.undistort_points(valid_cam_2d_points)
        # valid_cam_2d_points = np.round(valid_cam_2d_points).astype(int)
        for i, valid_index in enumerate(valid_2d_points):
            colours[valid_index, :] = pixel_colours[valid_cam_2d_points[i][1], valid_cam_2d_points[i][0], :]
        coloured_pcd = np.hstack([pcd, colours]) # [[x, y, z, I, R, G, B]]
        return coloured_pcd
