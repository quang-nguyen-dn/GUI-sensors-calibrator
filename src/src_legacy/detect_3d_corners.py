import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from PIL import Image, ImageEnhance
from scipy import io as sio
import cv2 as cv
import json
import matlab
import matlab.engine





def tf_matrix(R:np.ndarray=np.identity(3), t:np.ndarray=np.zeros((3))):
    if not isinstance(R, np.ndarray): R = np.array(R).reshape((3,3))
    if not isinstance(t, np.ndarray): t = np.array(t).reshape((3))
    result = np.identity(4)
    result[0:3, 0:3] = R
    result[0:3, 3] = t
    return result





def get_retina_corrected_image(image:np.ndarray, debug=False):
    retina = cv.bioinspired_Retina.create((image.shape[1], image.shape[0]))
    #retina.write('retinaParams.xml')
    retina.setup('retinaParams_grayscale.xml')
    retina.run(image)
    retina_parvo = retina.getParvo()
    retina_parvo = np.array(retina_parvo, dtype=np.uint8)
    if debug:
        fig, axs = plt.subplots(1, 3, figsize=(15,10))
        axs[0].imshow(image)
        axs[0].set_title('Original image')
        axs[1].imshow(retina_parvo)
        axs[1].set_title('Bio-retina corrected image')
        axs[2].imshow(retina.getMagno())
        raise RuntimeError
    return retina_parvo





def take_pointcloud_image(pcd_t, z_rotation:float, lidar_cam_intrinsics:np.ndarray=None):
    #pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd_o3d)

    if not isinstance(lidar_cam_intrinsics, np.ndarray):
        lidar_cam_width = 450
        lidar_cam_height = 300
        focal_length = 300
        #lidar_cam_intrinsics = np.array([
        #    [focal_length, 0, lidar_cam_width/2,],
        #    [0, focal_length, lidar_cam_height/2,],
        #    [0, 0, 1]
        #])
        lidar_cam_intrinsics_params = o3d.camera.PinholeCameraIntrinsic(
            width=lidar_cam_width,
            height=lidar_cam_height,
            fx=focal_length,
            fy=focal_length,
            cx=lidar_cam_width/2,
            cy=lidar_cam_height/2
        )
        lidar_cam_intrinsics = lidar_cam_intrinsics_params.intrinsic_matrix
    #print('Intrinsic:\n', lidar_cam_intrinsics)
    lidar_cam_extrinsics = np.identity(4)
    lidar_cam_extrinsics[0:3, 0:3] = Rotation.from_euler('XYZ', [90, 0, 90+z_rotation], degrees=True).as_matrix()
    lidar_cam_extrinsics[0:3, 3] = [0., 0., 0.]
    #print('Extrinsic:\n', lidar_cam_extrinsics)
    rgbd_img = pcd_t.project_to_rgbd_image(
        width=int(lidar_cam_intrinsics[0,2]*2), height=int(lidar_cam_intrinsics[1,2]*2),
        intrinsics=lidar_cam_intrinsics,
        extrinsics=lidar_cam_extrinsics,
        depth_max=10.0, depth_scale=1.0
    )
    #print(rgbd_img.color)

    intensity_img = np.asarray(rgbd_img.color, np.float32)
    #intensity_img = np.where(intensity_img >= 0.5, 1.0, 0.0)
    depth_img = np.asarray(rgbd_img.depth, np.float32)
    #print(depth_img.max())

    return intensity_img, depth_img






def detect_chessboard_from_partial_corners(depth_image:np.ndarray, corners_2D:np.ndarray, debug=False):
    corners_pixels = np.around(corners_2D).astype(int)
    i = corners_pixels[:, 1]
    j = corners_pixels[:, 0]
    z = depth_image[i, j].flatten()
    z_avg = np.average(z)






def detect_lidar_image_chessboard_corners(intensity_img:np.ndarray, matlab_engine, grid_inner_size=[6, 4], square_size=0.1, debug=False):
    #inp_img = np.array(intensity_img * 255.0, dtype=np.float32)
    #debug=True
    inp_img = np.array(intensity_img, dtype=np.float32)
    inp_img = inp_img[:, :, 0]
    inp_img_shape = inp_img.shape
    img_contrast = Image.fromarray(np.array(inp_img*255.0, dtype=np.uint8))
    #img_contrast = ImageEnhance.Contrast(img_contrast).enhance(10.0)
    #enhanced_img = np.array(img_contrast, np.float32) / 255.0
    enhanced_img = remove_intensity_outliers(inp_img.flatten(), ignore_zeros=True, debug=False).reshape(inp_img_shape)

    img = matlab.double(enhanced_img.tolist())
    chessboard_corners, board_size = matlab_engine.detectCheckerboardPoints(
        img, 'MinCornerMetric', 0.17, 'PartialDetections', False, nargout=2)
    chessboard_corners = np.array(chessboard_corners)

    if debug:
        try:
            fig, axs = plt.subplots(2, 3, figsize=(20,15), gridspec_kw={'width_ratios': [4, 1, 1]})
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
        except:
            pass
    
    if False:
    #if board_size[0][0] == 0 or board_size[0][1] == 0:
    #if board_size[0][1] != grid_inner_size[0]+2 or board_size[0][0] != grid_inner_size[1]+2:
        retina_img = get_retina_corrected_image(inp_img, debug=False) / 255.0

        img_contrast = Image.fromarray(np.array(retina_img*255.0, dtype=np.uint8))
        #img_contrast = ImageEnhance.Contrast(img_contrast).enhance(3.0)
        enhanced_retina_img = np.array(img_contrast, np.float32) / 255.0

        img = matlab.double(enhanced_retina_img.tolist())
        chessboard_corners, board_size = matlab_engine.detectCheckerboardPoints(
            img, 'MinCornerMetric', 0.10, 'PartialDetections', False, nargout=2)

        chessboard_corners = np.array(chessboard_corners)

        if debug:
            axs[1].set_title('Bio-retina enhanced intensity image')
            axs[1].imshow(enhanced_retina_img, cmap='gray')
            if len(chessboard_corners) > 0:
                axs[1].scatter(chessboard_corners[:,0], chessboard_corners[:,1], s=30, facecolors='none', edgecolors='r')
    
    return chessboard_corners, np.array([board_size[0][1], board_size[0][0]])





def depth_to_pcd(depth_image:np.ndarray, pixels_of_interest:np.ndarray, fx:float, fy:float, cx:float, cy:float, depth_cam_extrinsics:np.ndarray, remove_depth_outlier=False):
    pixels_of_interest = np.around(pixels_of_interest).astype(int)
    i = pixels_of_interest[:, 1]
    j = pixels_of_interest[:, 0]
    z = depth_image[i, j].flatten()

    neighbour_size = 5
    for index, z_i in enumerate(z):
        if z_i == 0.:
            for u in range(neighbour_size):
                for v in range(neighbour_size):
                    try:
                        dd = [
                            depth_image[i[index]+u, j[index]+v][0],
                            depth_image[i[index]+u, j[index]-v][0],
                            depth_image[i[index]-u, j[index]+v][0],
                            depth_image[i[index]-u, j[index]-v][0]
                        ]
                        for d in dd:
                            if d > 0:
                                z[index] = d
                                break
                    except:
                        pass

    if remove_depth_outlier:
        median_z = np.median(z)
        z = np.where(z > median_z + 0.2, median_z, z)

    i = i[z!=0]; j=j[z!=0]; z=z[z!=0]
    #print(result_pcd[:,2])
    x = 1/fx * ((j - cx) * z)
    y = 1/fy * ((i - cy) * z)
    
    result_pcd = np.vstack((x, y, z, np.ones(x.shape)))
    result_pcd = np.linalg.inv(depth_cam_extrinsics) @ result_pcd
    return np.transpose(result_pcd[0:3, :])





def orthogonal_projection(points:np.ndarray, plane_point:np.ndarray, plane_normal:np.ndarray):
    new_plane_normal = plane_normal / np.linalg.norm(plane_normal)
    #I = np.empty(points.shape)
    #I[:, 0:3] = plane_point
    #n = np.empty(points.shape)
    #n[:, 0:3] = plane_normal
    #print(I); raise RuntimeError
    v = points - plane_point
    temp = v * new_plane_normal
    dist = temp[:,0] + temp[:,1] + temp[:,2]
    dist = dist.reshape((-1,1))
    #print(dist)
    #print(n * dist)
    return points - dist * new_plane_normal






def bounded_3d_region_growing(
        pcd:o3d.geometry.PointCloud,
        init_centre:np.ndarray,
        init_R:np.ndarray,
        bbox_extent:list,
        convergent_distance:float,
):
    bbox = o3d.geometry.OrientedBoundingBox(
        center = init_centre,
        R = init_R,
        extent = bbox_extent
    )
    for num_iterations in range(50):
        prev_centre = bbox.center
        new_pcd = pcd.crop(bbox)
        new_pcd = new_pcd.voxel_down_sample(voxel_size=0.05)
        new_centre = np.average(np.array(new_pcd.points), axis=0)
        if np.linalg.norm(new_centre - prev_centre) <= convergent_distance:
            break
        else:
            bbox.center = prev_centre + 1.5*(new_centre - prev_centre)
    
    return bbox






def automatic_chessboard_detection(pcd, matlab_engine, grid_inner_size=[6, 4], square_size=0.1, debug=False):
    num_expected_corners = (grid_inner_size[0]+1) * (grid_inner_size[1]+1)

    lidar_cam_width = 250
    lidar_cam_height = 150
    focal_length = 450
    lidar_cam_intrinsics_params = o3d.camera.PinholeCameraIntrinsic(
        width=lidar_cam_width,
        height=lidar_cam_height,
        fx=focal_length,
        fy=focal_length,
        cx=lidar_cam_width/2,
        cy=lidar_cam_height/2
    )
    lidar_cam_intrinsics = lidar_cam_intrinsics_params.intrinsic_matrix
    pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd)
    
    chessboard_detected = False
    for z_rotation in range(-30, 330, 10):
        #print('Testing z_rotation =', z_rotation)
        intensity_img, depth_img = take_pointcloud_image(pcd_t, z_rotation, lidar_cam_intrinsics)
        chessboard_corners, board_size = detect_lidar_image_chessboard_corners(
            intensity_img, matlab_engine, grid_inner_size, square_size, debug=False
        )
        lidar_cam_extrinsics = tf_matrix(
            R = Rotation.from_euler('XYZ', [90, 0, 90+z_rotation], degrees=True).as_matrix()
        )
        if board_size[0] == grid_inner_size[0]+2 and board_size[1] == grid_inner_size[1]+2:
            #chessboard_2D_center = np.average(chessboard_corners, axis=0)
            #dist_to_image_center = np.abs(chessboard_2D_center - np.array([lidar_cam_width/2, lidar_cam_height/2], np.float32))
            #if dist_to_image_center[0] > lidar_cam_width/4 or dist_to_image_center[1] > lidar_cam_height/4:
            #    continue
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
                    # Confirmed that the chessboard is detected enrirely
                    est_chessboard_bbox.extent = [
                        (grid_inner_size[0]+8)*square_size, (grid_inner_size[1]+4)*square_size, 0.3
                    ]
                    chessboard_detected = True
                    #break
        
        elif 0 < board_size[0] <= grid_inner_size[0]+2 and 0 < board_size[1] <= grid_inner_size[1]+2:
            chessboard_2D_center = np.average(chessboard_corners, axis=0)
            dist_to_image_center = np.abs(chessboard_2D_center - np.array([lidar_cam_width/2, lidar_cam_height/2], np.float32))
            if dist_to_image_center[0] > lidar_cam_width * 7./8.:# or dist_to_image_center[1] > lidar_cam_height/10:
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
                detect_lidar_image_chessboard_corners(
                    intensity_img, matlab_engine, grid_inner_size, square_size, debug=debug
                )
            #est_chessboard_bbox.extent = [
            #    (grid_inner_size[0]+8)*square_size, (grid_inner_size[1]+4)*square_size, 0.3
            #]
            #est_chessboard_bbox.color = [0,0,1]

            '''for num_iterations in range(50):
                prev_center = est_chessboard_bbox.center
                new_chessboard_pcd = pcd.crop(est_chessboard_bbox)
                new_chessboard_pcd = new_chessboard_pcd.voxel_down_sample(voxel_size=square_size/2)
                #new_chessboard_bbox = new_chessboard_pcd.get_minimal_oriented_bounding_box(robust=True)
                #new_center = new_chessboard_bbox.center
                new_center = np.average(np.array(new_chessboard_pcd.points), axis=0)
                if np.linalg.norm(new_center - prev_center) <= square_size/10:
                    break
                else:
                    est_chessboard_bbox.center = new_center'''
            
            est_chessboard_bbox = bounded_3d_region_growing(
                pcd,
                est_chessboard_bbox.center,
                est_chessboard_bbox.R,
                [(grid_inner_size[0]+6)*square_size, (grid_inner_size[1]+3)*square_size, 0.3],
                square_size/10.
            )

            if debug:
                est_chessboard_bbox.color = np.array([16., 122., 28.]) / 255.
                debug_arrow = o3d.geometry.TriangleMesh.create_coordinate_frame()
                debug_arrow.transform(tf_matrix(est_chessboard_bbox.R, est_chessboard_bbox.center))
                est_corners_pcd.paint_uniform_color([1,0,0])
                try:
                    o3d.visualization.draw_geometries([pcd, est_corners_pcd, est_chessboard_bbox, partial_bbox, debug_arrow])
                except:
                    o3d.visualization.draw_geometries([pcd, est_corners_pcd, est_chessboard_bbox, debug_arrow])
            #est_chessboard_bbox.extent = [
            #    (grid_inner_size[0]+3)*square_size, (grid_inner_size[1]+3)*square_size, 0.3
            #]
            break


                    

    #if chessboard_corners.shape[0] != num_expected_corners:
    try:
        est_corners_pcd
    except:
        raise ValueError('Could not detect calibration target from input point cloud')

    #lidar_cam_extrinsics = np.identity(4)
    #lidar_cam_extrinsics[0:3, 0:3] = Rotation.from_euler('XYZ', [90, 0, 90+z_rotation], degrees=True).as_matrix()
    #lidar_cam_extrinsics[0:3, 3] = [0, 0, 0]
    
    #corners_depths = corners_3d_estimates[:,2]
    #median_corners_depth = np.median(corners_depths)
    #corners_3d_estimates[:,2] = np.where(corners_depths > median_corners_depth+0.05, median_corners_depth, corners_depths)

    #est_centroid = np.average(corners_3d_estimates, axis=0)
    #print(est_centroid)
    #est_eigvals, est_eigvects = np.linalg.eig(np.transpose(corners_3d_estimates-est_centroid) @ (corners_3d_estimates-est_centroid))
    #est_eigvals_order = np.argsort(est_eigvals)[::-1]
    #est_eigvals = est_eigvals[est_eigvals_order]
    #est_eigvects = est_eigvects[:, est_eigvals_order]
    #print(est_eigvals, np.sqrt(est_eigvals)); print(est_eigvects)
    
    
    #est_chessboard_bbox = o3d.geometry.OrientedBoundingBox(
    #    center = est_centroid,
    #    R = Rotation.align_vectors([est_eigvects[:,0], est_eigvects[:,1]], [[1,0,0], [0,1,0]])[0].as_matrix(),
    #    extent=[(grid_inner_size[0]+3)*square_size, (grid_inner_size[1]+3)*square_size, square_size*2]
    #    #extent=[0.2, 1.6, 1.2]
    #)
    #est_corners_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(corners_3d_estimates))
    est_corners_pcd.paint_uniform_color([1,0,0])
    #est_chessboard_bbox = est_corners_pcd.get_oriented_bounding_box()
    #est_chessboard_bbox.extent = [(grid_inner_size[0]+1.5)*square_size, (grid_inner_size[1]+1.5)*square_size, 0.2]
    est_chessboard_bbox.color = np.array([16., 122., 28.]) / 255.

    rough_chessboard_pcd = pcd.crop(est_chessboard_bbox)
    _, chessboard_plane_inliers = rough_chessboard_pcd.segment_plane(
        distance_threshold=0.04,
        ransac_n=30,
        num_iterations=100
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

    rough_chessboard_pcd.paint_uniform_color([0,0,1])
    chessboard_inliers_pcd.paint_uniform_color([0,1,0])
   
    #o3d.visualization.draw_geometries([pcd, chessboard_bbox])

    #o3d.visualization.draw_geometries([chessboard_pcd, chessboard_bbox])
    chessboard_pcd_intensities = np.array(chessboard_pcd.colors)[:,0]
    #print(chessboard_pcd_intensities.shape, np.min(chessboard_pcd_intensities))
    #chessboard_pcd_intensities -= np.min(chessboard_pcd_intensities)
    #chessboard_pcd_intensities /= np.max(chessboard_pcd_intensities)
    chessboard_pcd_intensities = remove_intensity_outliers(chessboard_pcd_intensities, debug=False)

    chessboard_3d_points = np.array(chessboard_pcd.points)    
    chessboard_pose = tf_matrix(chessboard_bbox.R, centroid)

    chessboard_bbox.color = np.array([1, 0, 0], np.float64)
    if debug:
        #print(corners_3d_estimates)
        chessboard_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)
        chessboard_coord_frame.transform(chessboard_pose)
        world_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)
        o3d.visualization.draw_geometries([
            rough_chessboard_pcd, chessboard_inliers_pcd, chessboard_pcd, est_corners_pcd, chessboard_bbox, 
            world_coord_frame, est_chessboard_bbox
        ])
        #raise KeyboardInterrupt

    return chessboard_3d_points, chessboard_pcd_intensities, centroid, chessboard_x_axis, chessboard_y_axis, chessboard_z_axis





def get_combined_chessboard_3D_points(
        pcds_batch: 'dict[o3d.geometry.PointCloud]',
        lidar_extrinsics: 'dict[np.ndarray]',
        init_chessboard_pose: np.ndarray,
        init_lidar_name = 'os128',
        grid_inner_size = [6, 4],
        square_size = 0.1,
        debug=False
):
    combined_chessboard_pcd = o3d.geometry.PointCloud()
    combined_chessboard_intensities = np.array([], dtype=np.float32)
    bbox_extent = np.array([(grid_inner_size[0]+4)*square_size, (grid_inner_size[1]+4)*square_size, 1.0])
    init_chessboard_pose = init_chessboard_pose.copy()
    init_chessboard_pose = lidar_extrinsics[init_lidar_name] @ init_chessboard_pose
    init_chessboard_bbox = o3d.geometry.OrientedBoundingBox(
        center = init_chessboard_pose[0:3, 3],
        R = init_chessboard_pose[0:3, 0:3],
        extent = bbox_extent
    )
    init_pcd = o3d.geometry.PointCloud(pcds_batch[init_lidar_name].points)
    init_pcd.transform(lidar_extrinsics[init_lidar_name])
    init_pcd.estimate_normals()
    init_pcd.paint_uniform_color([0,0,0])
    init_cb_arrows = o3d.geometry.TriangleMesh.create_coordinate_frame()
    init_cb_arrows.transform(init_chessboard_pose)
    chessboards_poses = {lidar_name: None for lidar_name in pcds_batch.keys()}
    chessboards_poses[init_lidar_name] = init_chessboard_pose
    #o3d.visualization.draw_geometries([pcds_batch['os128'], init_chessboard_bbox])
    refined_lidar_extrinsics = {}
    num_processed_pcds = 0
    for lidar_name in pcds_batch.keys():
        print('Refining extrinsics of LiDAR', lidar_name)
        if lidar_name == init_lidar_name:
            refined_lidar_extrinsics[init_lidar_name] = lidar_extrinsics[init_lidar_name]
            print('Done refining extrinsics for LiDAR', lidar_name)
            continue
        #if lidar_name != 'os64': continue

        # Consider chessboard pose in VLS128 coord frame
        current_chessboard_pose = np.linalg.inv(lidar_extrinsics[lidar_name]) @ init_chessboard_pose
        current_chessboard_rough_bbox = o3d.geometry.OrientedBoundingBox(
            center = current_chessboard_pose[0:3, 3],
            R = current_chessboard_pose[0:3, 0:3],
            extent = bbox_extent
        )
        current_chessboard_rough_bbox.color = [1, 0, 0]
        #o3d.visualization.draw_geometries([pcds_batch[lidar_name], current_chessboard_bbox])
        #raise RuntimeError
        current_chessboard_pcd = pcds_batch[lidar_name].crop(current_chessboard_rough_bbox)
        _, chessboard_plane_inliers = current_chessboard_pcd.segment_plane(
            distance_threshold=0.05,
            ransac_n=30,
            num_iterations=200
        )
        current_chessboard_pcd = current_chessboard_pcd.select_by_index(chessboard_plane_inliers)

        #combined_chessboard_intensities = np.append(combined_chessboard_intensities, I)
        #combined_chessboard_pcd += current_chessboard_pcd.transform(lidar_extrinsics[lidar_name])
        #continue

        current_chessboard_bbox = current_chessboard_pcd.get_minimal_oriented_bounding_box(robust=True)
        centroid = np.array(current_chessboard_bbox.center)
        world_axes = np.identity(3)
        ax_order = np.argsort(current_chessboard_bbox.extent)[::-1]

        chessboard_x_axis = current_chessboard_bbox.R @ world_axes[ax_order[0]]
        chessboard_y_axis = current_chessboard_bbox.R @ world_axes[ax_order[1]]
        if np.dot(chessboard_y_axis, [0., 0., 1.]) < 0:
            chessboard_y_axis = -chessboard_y_axis
            chessboard_x_axis = -chessboard_x_axis
        chessboard_z_axis = np.cross(chessboard_x_axis, chessboard_y_axis)
        if np.dot(chessboard_z_axis, centroid) > 0:
            chessboard_x_axis = -chessboard_x_axis
            chessboard_z_axis = -chessboard_z_axis
        
        expanded_chessboard_bbox = bounded_3d_region_growing(
            pcds_batch[lidar_name],
            centroid,
            Rotation.align_vectors([chessboard_x_axis, chessboard_y_axis], [[1,0,0], [0,1,0]])[0].as_matrix(),
            [(grid_inner_size[0]+6)*square_size, (grid_inner_size[1]+4)*square_size, 0.3],
            square_size/10
        )
        expanded_chessboard_bbox.color = [1,0,0]
        #expanded_chessboard_bbox.extent = [(grid_inner_size[0]+2)*square_size, (grid_inner_size[1]+3)*square_size, 0.3]
        rough_chessboard_pcd = pcds_batch[lidar_name].crop(expanded_chessboard_bbox)

        _, chessboard_plane_inliers = rough_chessboard_pcd.segment_plane(
            distance_threshold=0.04,
            ransac_n=10,
            num_iterations=100
        )
        chessboard_inliers_pcd = rough_chessboard_pcd.select_by_index(chessboard_plane_inliers)
        chessboard_bbox = chessboard_inliers_pcd.get_minimal_oriented_bounding_box(robust=True)

        # Orientation correction
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
            (grid_inner_size[0]+2)*square_size, (grid_inner_size[1]+2)*square_size, 0.04
        ]
        current_chessboard_pcd = chessboard_inliers_pcd.crop(chessboard_bbox)

        I = np.array(current_chessboard_pcd.colors)[:,0]
        I = remove_intensity_outliers(I, debug=debug)
        
        current_chessboard_pcd.paint_uniform_color([1,0,0])
        #o3d.visualization.draw_geometries([pcds_batch[lidar_name], current_chessboard_pcd, expanded_chessboard_bbox])
        #raise RuntimeError
        
        current_opt_tf = optimise_chessboard_tf(
            np.array(current_chessboard_pcd.points), I, centroid,
            chessboard_x_axis, chessboard_y_axis, chessboard_z_axis,
            grid_inner_size, square_size,
            debug=False
        )
        current_opt_pose = get_chessboard_pose(
            current_opt_tf, centroid, chessboard_x_axis, chessboard_y_axis, chessboard_z_axis
        )
        num_processed_pcds += 1

        chessboards_poses[lidar_name] = current_opt_pose

        current_pcd = o3d.geometry.PointCloud(pcds_batch[lidar_name].points)

        init_tf = init_chessboard_pose @ np.linalg.inv(current_opt_pose)
        init_tf[0:3, 0:3] = lidar_extrinsics[lidar_name][0:3, 0:3]
        current_pcd.transform(init_tf)
        current_pcd.paint_uniform_color(np.array([237., 135., 26.])/255.)
        current_pcd.estimate_normals()
        current_chessboard_pcd.transform(init_chessboard_pose @ np.linalg.inv(current_opt_pose))
        #current_chessboard_pcd.transform(lidar_extrinsics[lidar_name])
        cb_pose_arrows_orig = o3d.geometry.TriangleMesh.create_coordinate_frame()
        cb_pose_arrows_orig.transform(lidar_extrinsics[lidar_name] @ current_opt_pose)

        if False:
            current_pcd_orig_extrinsics = o3d.geometry.PointCloud(pcds_batch[lidar_name].points)
            current_pcd_orig_extrinsics.transform(lidar_extrinsics[lidar_name])
            current_pcd_orig_extrinsics.paint_uniform_color(np.array([237., 135., 26.])/255.)
            o3d.visualization.draw_geometries([init_pcd, current_pcd_orig_extrinsics, cb_pose_arrows_orig, init_cb_arrows])

        cb_pose_arrows = o3d.geometry.TriangleMesh.create_coordinate_frame()
        cb_pose_arrows.transform(init_tf @ current_opt_pose)

        if True:
            icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration = 1000,
                relative_fitness = 1e-9,
                relative_rmse = 1e-9
            )
            GICP = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(
                epsilon=1e-6
            )

            crop_xpositive_bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound = [0, -100, -100],
                max_bound = [100, 100, 100]
            )

            refined_matching_result = o3d.pipelines.registration.registration_icp(
                source = current_pcd.crop(crop_xpositive_bbox),#.voxel_down_sample(0.1),
                target = init_pcd.crop(crop_xpositive_bbox),#.voxel_down_sample(0.1),
                max_correspondence_distance = 1.0,
                #init = np.identity(4),
                #estimation_method = GICP,
                estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria = icp_criteria
            )
            print(refined_matching_result)
            current_pcd.transform(refined_matching_result.transformation)
            current_chessboard_pcd.transform(refined_matching_result.transformation)
            cb_pose_arrows.transform(refined_matching_result.transformation)

        if debug:
            o3d.visualization.draw_geometries([init_pcd, current_pcd, cb_pose_arrows, init_cb_arrows])
        #raise RuntimeError
        refined_lidar_extrinsics[lidar_name] = refined_matching_result.transformation @ init_tf
        print('Done refining extrinsics for LiDAR', lidar_name)

    #combined_chessboard_pcd.colors = o3d.utility.Vector3dVector(
    #    plt.get_cmap('gray')(combined_chessboard_intensities)[:, 0:3]
    #)

    #_, chessboard_plane_inliers = combined_chessboard_pcd.segment_plane(
    #    distance_threshold=0.04,
    #    ransac_n=20,
    #    num_iterations=100
    #)
    #refined_combined_chessboard_pcd = combined_chessboard_pcd.select_by_index(chessboard_plane_inliers)
    combined_pcd = o3d.geometry.PointCloud()
    for i, lidar_name in enumerate(pcds_batch.keys()):
        try:
            tf = refined_lidar_extrinsics[lidar_name]
        except:
            continue
        temp_pcd = o3d.geometry.PointCloud(pcds_batch[lidar_name])
        temp_pcd.transform(tf)
        temp_pcd.paint_uniform_color(plt.get_cmap('tab10')(i)[:3])
        combined_pcd += temp_pcd

    if debug:        
        o3d.visualization.draw_geometries([combined_pcd])
    
    return refined_lidar_extrinsics, combined_pcd






def find_optimised_chessboard_3D_corners(
        pcd_o3d:o3d.geometry.PointCloud,
        grid_inner_size=[6,4],
        square_size=0.1
):
    I = np.array(pcd_o3d.colors)[:,0]
    bbox = pcd_o3d.get_minimal_oriented_bounding_box(robust=True)
    centroid = np.array(bbox.center)
    world_axes = np.identity(3)
    ax_order = np.argsort(bbox.extent)[::-1]

    chessboard_x_axis = bbox.R @ world_axes[ax_order[0]]
    chessboard_y_axis = bbox.R @ world_axes[ax_order[1]]
    if np.dot(chessboard_y_axis, [0., 0., 1.]) < 0:
        chessboard_y_axis = -chessboard_y_axis
        chessboard_x_axis = -chessboard_x_axis
    chessboard_z_axis = np.cross(chessboard_x_axis, chessboard_y_axis)
    if np.dot(chessboard_z_axis, centroid) > 0:
        chessboard_x_axis = -chessboard_x_axis
        chessboard_z_axis = -chessboard_z_axis

    opt_tf = optimise_chessboard_tf(
        np.array(pcd_o3d.points), I, centroid,
        chessboard_x_axis, chessboard_y_axis, chessboard_z_axis,
        grid_inner_size, square_size
    )

    opt_pose = get_chessboard_pose(
        opt_tf, centroid, chessboard_x_axis, chessboard_y_axis, chessboard_z_axis
    )
    return opt_pose





def save_chessboard_pcd(
        chessboard_pose:np.ndarray,
        orig_pcd:o3d.geometry.PointCloud,
        filename:str,
        grid_inner_size = [6,4],
        square_size=0.1
):
    chessboard_bbox = o3d.geometry.OrientedBoundingBox(
        center = chessboard_pose[:3,3],
        R = chessboard_pose[:3,:3],
        extent = [0.8, 0.6, 0.1]
        #extent = [1.2, 0.7, 0.1]
    )
    legacy_chessboard_pcd = orig_pcd.crop(chessboard_bbox)
    points = np.array(legacy_chessboard_pcd.points)
    #print(points); raise RuntimeError
    intensities = np.array(legacy_chessboard_pcd.colors)[:,0].reshape((-1,1))
    intensities = remove_intensity_outliers(intensities)
    chessboard_pcd = o3d.t.geometry.PointCloud.from_legacy(legacy_chessboard_pcd)
    del chessboard_pcd.point.colors
    #chessboard_pcd.point.positions = o3d.core.Tensor(points)
    chessboard_pcd.point.intensity = o3d.core.Tensor(intensities)
    o3d.t.io.write_point_cloud(filename, chessboard_pcd)







def get_chessboard_2d_projection(
        chessboard_3d:np.ndarray, centroid:np.ndarray, x_axis:np.ndarray, y_axis:np.ndarray, z_axis:np.ndarray,
        #x_rot:float=0., y_rot:float=0., z_rot:float=0., u_trans:float=0., v_trans:float=0.,
        chessboard_tf:np.ndarray=np.zeros(5),
        grid_inner_size=[6,4], square_size=0.1
    ):
    
    x_rot, y_rot, z_rot, u_trans, v_trans = chessboard_tf
    chessboard_rotation = Rotation.align_vectors([x_axis, y_axis], [[1,0,0], [0,1,0]])[0].as_matrix()
    rotation_adjustment = Rotation.from_euler('XYZ', [x_rot, y_rot, z_rot], degrees=True).as_matrix()
    projected_3d_chessboard = orthogonal_projection(chessboard_3d, centroid, z_axis)
    projected_3d_chessboard -= centroid
    projected_3d_chessboard = rotation_adjustment @ np.linalg.inv(chessboard_rotation) @ np.transpose(projected_3d_chessboard)
    projected_3d_chessboard = np.transpose(projected_3d_chessboard)
    #print(projected_3d_chessboard)
    chessboard_uv = projected_3d_chessboard[:, 0:2]
    chessboard_uv -= np.array([u_trans, v_trans])
        
    return chessboard_uv





def get_chessboard_2d_projection_alt(
        pcd:o3d.geometry.PointCloud, centroid:np.ndarray, x_axis:np.ndarray, y_axis:np.ndarray, z_axis:np.ndarray,
        #x_rot:float=0., y_rot:float=0., z_rot:float=0., u_trans:float=0., v_trans:float=0.
        chessboard_tf:np.ndarray=np.zeros(5),
        grid_inner_size=[6,4], square_size=0.1
):
    new_chessboard_pose = get_chessboard_pose(chessboard_tf, centroid, x_axis, y_axis, z_axis)
    chessboard_bbox = o3d.geometry.OrientedBoundingBox(
        center = new_chessboard_pose[0:3, 3],
        R = new_chessboard_pose[0:3, 0:3],
        extent=[(grid_inner_size[0]+3)*square_size, (grid_inner_size[1]+3)*square_size, square_size*2]
    )
    chessboard_pcd_o3d = pcd.crop(chessboard_bbox)
    #o3d.visualization.draw_geometries([pcd, chessboard_bbox])
    chessboard_pcd = np.array(chessboard_pcd_o3d.points)
    chessboard_intensity = np.array(chessboard_pcd_o3d.colors)[:,0]
    chessboard_intensity -= np.min(chessboard_intensity)
    chessboard_intensity /= np.max(chessboard_intensity)

    projected_3d_chessboard = orthogonal_projection(
        chessboard_pcd, 
        new_chessboard_pose[0:3, 3],
        new_chessboard_pose[0:3, 0:3] @ np.array([0., 0., 1.])
    )
    projected_3d_chessboard -= new_chessboard_pose[0:3, 3]
    projected_3d_chessboard = np.transpose(np.linalg.inv(new_chessboard_pose[0:3, 0:3]) @ np.transpose(projected_3d_chessboard))
    chessboard_uv = projected_3d_chessboard[:, 0:2]
    return chessboard_uv, chessboard_intensity






def get_chessboard_square_id(chessboard_uv:np.ndarray, grid_inner_size=np.array([6,4], dtype=np.int8), square_size=0.1):
    grid_size = grid_inner_size + np.array([2, 2])
    chessboard_uv_ids = chessboard_uv/square_size + grid_size/2
    #chessboard_uv_ids = np.trunc(chessboard_uv_ids).astype(np.int8)
    #chessboard_uv_ids[chessboard_uv[:,0]>=grid_size[0]/2*square_size] += np.array([-1, 0], np.int8)
    #print(chessboard_uv_ids)
    def sqid(uv_row):
        if uv_row[0] < 0 or uv_row[0] >= grid_size[0]: return -1
        if uv_row[1] < 0 or uv_row[1] >= grid_size[1]: return -1
        return np.trunc(uv_row[0]) + grid_size[0]*np.trunc(uv_row[1])
    chessboard_square_ids = np.apply_along_axis(sqid, 1, chessboard_uv_ids)
    chessboard_square_ids = chessboard_square_ids.astype(np.int8)
    #chessboard_square_ids[~valid_u, 0] = -1
    #chessboard_square_ids[~valid_v, 1] = -1
    #print(np.where(chessboard_square_ids>23)[0])
    return chessboard_square_ids





def determine_squares_colours(chessboard_intensities:np.ndarray, square_ids:np.ndarray, grid_inner_size=[6,4]):
    grid_size = np.array(grid_inner_size) + np.array([2., 2.])
    num_squares = np.max(square_ids) + 1
    square_colours = np.zeros((num_squares), np.uint8)
    for i in range(num_squares):
        avg_square_intensity = np.average(chessboard_intensities[square_ids==i])
        if avg_square_intensity >= 0.5: square_colours[i] = 1
    
    square_0_family = np.arange(0, num_squares, 2, dtype=int)
    row_ids = np.trunc(square_0_family / grid_size[0])
    square_0_family[row_ids % 2 == 1] += 1
    square_0_family = np.in1d(range(num_squares), square_0_family)

    square_colours[square_0_family] = 1.0
    square_colours[~square_0_family] = 0.0

    #print(square_0_family)
    #print('Avg colours of square 0 family:', square_colours[square_0_family])
    #print('Avg colours of square 1 family:', square_colours[~square_0_family])

    '''
    # Check if the square 0 family is white square
    num_white_square_0_family = np.sum(square_colours[square_0_family])
    num_white_square_1_family = np.sum(square_colours[~square_0_family])
    num_black_square_0_family = num_squares//4 - num_white_square_0_family
    num_black_square_1_family = num_squares//4 - num_white_square_1_family
    #if num_white_square_0_family > num_white_square_1_family \
    #    and num_black_square_0_family <= num_black_square_1_family:
    if num_white_square_0_family > num_white_square_1_family:
            # Square 0 family is white
            square_colours[square_0_family] = 1
            square_colours[~square_0_family] = 0
    #elif num_white_square_0_family <= num_white_square_1_family \
    #    and num_black_square_0_family > num_black_square_1_family:
    else:
            # Square 0 family is black
            square_colours[square_0_family] = 0
            square_colours[~square_0_family] = 1
    #else:
    #    raise ValueError
    '''

    #print('Colours of square 0 family:', square_colours[square_0_family])
    #print('Colours of square 1 family:', square_colours[~square_0_family])

    return square_colours





def get_square_uv(grid_inner_size=[6,4], square_size=0.1):
    grid_inner_size = np.array(grid_inner_size)
    grid_size = grid_inner_size + np.array([2, 2])
    squares_uv = np.zeros((grid_size[1], grid_size[0], 2))
    for row_id in range(grid_size[1]):
        for col_id in range(grid_size[0]):
            squares_uv[row_id, col_id] = [col_id*square_size, row_id*square_size]
    squares_uv -= grid_size / 2 * square_size
    squares_uv += square_size/2
    squares_uv = squares_uv.reshape((-1,2))
    return squares_uv





def remove_intensity_outliers(intensities:np.ndarray, ignore_zeros=False, debug=False):
    orig_I = intensities.copy()
    if ignore_zeros:
        valid_ids = np.where(orig_I > 0)[0]
    else:
        valid_ids = np.arange(orig_I.shape[0], dtype=int)

    if len(valid_ids) == 0:
        return orig_I

    orig_I = (orig_I - np.min(orig_I)) / (np.max(orig_I) - np.min(orig_I))
    I = orig_I[valid_ids]
    

    while True:
        Q1 = np.percentile(I, 25)
        Q3 = np.percentile(I, 75)
        IQR = Q3 - Q1
        IQR_range = 1.0
        min_inlier = Q1 - IQR_range*IQR
        max_inlier = Q3 + IQR_range*IQR

        #avg = np.mean(I)
        #std = np.std(I)
        #min_inlier = avg - 2*std
        #max_inlier = avg + 2*std

        lower_outliers = np.where(I < min_inlier)[0]
        upper_outliers = np.where(I > max_inlier)[0]
        inliers = np.logical_and(I >= min_inlier, I <= max_inlier)

        if len(lower_outliers) == 0 and len(upper_outliers) == 0:
            break

        I[lower_outliers] = 0.
        I[upper_outliers] = 1.
        I[inliers] = (I[inliers] - np.min(I[inliers])) / (np.max(I[inliers]) - np.min(I[inliers]))
        break

    orig_I[valid_ids] = I

    if debug:
        fig, axs = plt.subplots(1, 4, figsize=(10, 5))
        axs[0].boxplot(intensities[valid_ids], whis=IQR_range)
        axs[1].hist(intensities[valid_ids], bins=50, orientation='horizontal')
        axs[2].boxplot(I, whis=IQR_range)
        axs[3].hist(I, bins=50, orientation='horizontal')
    return orig_I





def chessboard_cost_function(
    input_x:np.ndarray,
    pcd:o3d.geometry.PointCloud, 
    orig_chessboard_intensities:np.ndarray,
    centroid:np.ndarray, 
    x_axis:np.ndarray, y_axis:np.ndarray, z_axis:np.ndarray,
    #x_rot:float=0., y_rot:float=0., z_rot:float=0., u_trans:float=0., v_trans:float=0.,
    grid_inner_size=[6,4], square_size=0.1, debug=False,
    temp_dir:str=''
):
    grid_inner_size = np.array(grid_inner_size)
    num_squares = (grid_inner_size[0]+2) * (grid_inner_size[1]+2)
    #x_rot, y_rot, z_rot, u_trans, v_trans = input_x
    #orig_chessboard_uv, orig_chessboard_intensities = get_chessboard_2d_projection(
    #    pcd, centroid, x_axis, y_axis, z_axis,
    #    input_x,
    #    grid_inner_size, square_size
    #)
    orig_chessboard_uv = get_chessboard_2d_projection(
        pcd, centroid, x_axis, y_axis, z_axis,
        input_x,
        grid_inner_size, square_size
    )
    orig_chessboard_square_ids = get_chessboard_square_id(orig_chessboard_uv, grid_inner_size, square_size)
    valid_square_ids = np.where(orig_chessboard_square_ids != -1)[0]
    N = len(valid_square_ids)
    chessboard_uv = orig_chessboard_uv[valid_square_ids]
    chessboard_square_ids = orig_chessboard_square_ids[valid_square_ids]
    chessboard_intensities = orig_chessboard_intensities[valid_square_ids]
    #chessboard_intensities = remove_intensity_outliers(chessboard_intensities)

    squares_uv = get_square_uv(grid_inner_size, square_size)
    chessboard_square_uv = np.zeros_like(chessboard_uv)
    for i in range(len(chessboard_square_uv)):
        chessboard_square_uv[i] = squares_uv[chessboard_square_ids[i]]

    squares_colours = determine_squares_colours(chessboard_intensities, chessboard_square_ids, grid_inner_size)
    #print(squares_colours)
    #chessboard_square_intensities_1 = np.zeros_like(chessboard_intensities)
    #chessboard_square_intensities_2 = np.zeros_like(chessboard_intensities)
    #for i in range(len(chessboard_square_intensities_1)):
    #    chessboard_square_intensities_1[i] = squares_colours[chessboard_square_ids[i]]
    #    chessboard_square_intensities_2[i] = 1.0 - squares_colours[chessboard_square_ids[i]]
    
    chessboard_square_intensities = np.zeros_like(chessboard_intensities)
    for i in range(len(chessboard_square_intensities)):
        chessboard_square_intensities[i] = squares_colours[chessboard_square_ids[i]]
        
    dist_sq = (chessboard_uv[:,0] - chessboard_square_uv[:,0])**2 + (chessboard_uv[:,1] - chessboard_square_uv[:,1])**2
    dist = np.sqrt(dist_sq)
    I_coef = 10.0
    common_cost = (1.0 + dist/square_size)**2 + I_coef*(1.0 + chessboard_intensities - chessboard_square_intensities)**2
    #cost_1 = common_cost + I_coef*(1.0 + chessboard_intensities - chessboard_square_intensities_1)**2
    #cost_2 = common_cost + I_coef*(1.0 + chessboard_intensities - chessboard_square_intensities_2)**2

    def find_cost_per_square(cost_arr:np.ndarray):
        result = 0
        for square_id in range(num_squares):
            points_ids_in_square = np.where(chessboard_square_ids==square_id)[0]
            if len(points_ids_in_square) > 0:
                result += np.sum(cost_arr[points_ids_in_square])**2 / len(points_ids_in_square)
            else:
                continue
            #plt.scatter(chessboard_uv[points_ids_in_square, 0], chessboard_uv[points_ids_in_square, 1]); raise RuntimeError
            #rint(square_id, np.sum(cost_arr[points_ids_in_square]), len(points_ids_in_square), result)
        return result
    
    #costs = [np.sum(cost_1), np.sum(cost_2)]
    #costs = [find_cost_per_square(cost_1), find_cost_per_square(cost_2)]
    #cost_min_id = np.argmin(costs)
    #cost = costs[cost_min_id] #/ np.sqrt(N)
    cost = find_cost_per_square(common_cost) / N

    if debug:
        print(cost)
        fig, axs = plt.subplots(1, 2, figsize=(20,6))
        axs[0].set_facecolor([0,0.5,0])
        axs[0].scatter(chessboard_uv[:,0], chessboard_uv[:,1], c=chessboard_intensities, s=10, cmap='gray', vmin=0, vmax=1)
        axs[0].axis('equal')
        axs[0].grid()
        axs[0].set_title('cost = {:.3f}'.format(cost))
        axs[0].set_xlabel('u [m]')
        axs[0].set_ylabel('v [m]')
        axs[0].set_xlim(np.array([-grid_inner_size[0]/2-1, grid_inner_size[0]/2+1])*square_size)
        axs[0].set_ylim(np.array([-grid_inner_size[1]/2-1, grid_inner_size[1]/2+1])*square_size)
        axs[1].set_facecolor([0,0.5,0])
        #if cost_min_id == 0:
        #    axs[1].scatter(chessboard_uv[:,0], chessboard_uv[:,1], c=chessboard_square_intensities_1, s=10, cmap='gray')
        #else:
        #    axs[1].scatter(chessboard_uv[:,0], chessboard_uv[:,1], c=chessboard_square_intensities_2, s=10, cmap='gray')
        axs[1].scatter(chessboard_uv[:,0], chessboard_uv[:,1], c=chessboard_square_intensities, s=10, cmap='gray')
        axs[1].axis('equal')
        axs[1].grid()
        axs[1].set_title('Square intensities')
        axs[1].set_xlabel('u [m]')
        axs[1].set_ylabel('v [m]')
        axs[1].set_xlim(np.array([-grid_inner_size[0]/2-1, grid_inner_size[0]/2+1])*square_size)
        axs[1].set_ylim(np.array([-grid_inner_size[1]/2-1, grid_inner_size[1]/2+1])*square_size)
        axs[0].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(square_size))
        axs[1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(square_size))
        #axs[2].set_facecolor([0,0.5,0])
        #axs[2].scatter(chessboard_square_uv[:,0], chessboard_square_uv[:,1], c=chessboard_square_intensities, s=10, cmap='gray')
        #axs[2].axis('equal')
        #axs[2].grid()
    
    return cost






def optimise_chessboard_tf(
    pcd:o3d.geometry.PointCloud, orig_chessboard_intensities:np.ndarray, centroid:np.ndarray, 
    x_axis:np.ndarray, y_axis:np.ndarray, z_axis:np.ndarray,
    grid_inner_size=[6,4], square_size=0.1, 
    upper_bounds:np.ndarray = None, debug=False
):
    grid_inner_size = np.array(grid_inner_size)
    if not isinstance(upper_bounds, np.ndarray):
        upper_bounds = np.array([0., 0., 10., 0.9*square_size,0.9*square_size])
    result = scipy.optimize.minimize(
        fun = chessboard_cost_function,
        x0 = np.zeros(5),
        method = 'Powell',
        bounds = scipy.optimize.Bounds(-upper_bounds, upper_bounds),
        args = (
            pcd, orig_chessboard_intensities, centroid,
            #pcd, centroid,
            x_axis, y_axis, z_axis, grid_inner_size, square_size
        ),
        options = {
            'ftol': 1e-9,
        }
    )
    if debug: 
        print(result)
        chessboard_cost_function(
            np.zeros(5),
            pcd, orig_chessboard_intensities,
            centroid,
            x_axis, y_axis, z_axis, grid_inner_size, square_size, debug=True
        )

    opt_cost = chessboard_cost_function(
        result.x,
        pcd, orig_chessboard_intensities,
        centroid,
        x_axis, y_axis, z_axis, grid_inner_size, square_size, debug=True
    )
    #print('Optimised cost:', opt_cost)
    return result.x





def get_chessboard_pose(
        chessboard_tf:np.ndarray, centroid:np.ndarray, x_axis:np.ndarray, y_axis:np.ndarray, z_axis:np.ndarray,
):
    x_rot, y_rot, z_rot, u_trans, v_trans = chessboard_tf
    
    new_x_axis = x_axis / np.linalg.norm(x_axis)
    new_y_axis = y_axis / np.linalg.norm(y_axis)
    new_z_axis = z_axis / np.linalg.norm(z_axis)
    rot_mat = Rotation.from_rotvec(-x_rot/180.0*np.pi * new_x_axis).as_matrix() @\
              Rotation.from_rotvec(-y_rot/180.0*np.pi * new_y_axis).as_matrix() @\
              Rotation.from_rotvec(-z_rot/180.0*np.pi * new_z_axis).as_matrix()
    
    new_x_axis = rot_mat @ new_x_axis
    new_y_axis = rot_mat @ new_y_axis
    new_z_axis = rot_mat @ new_z_axis

    new_centroid = centroid + u_trans*new_x_axis + v_trans*new_y_axis
    chessboard_pose = tf_matrix(Rotation.align_vectors([new_x_axis, new_y_axis], [[1,0,0], [0,1,0]])[0].as_matrix(), new_centroid)
    return chessboard_pose





def get_chessboard_corners_3d(chessboard_pose:np.ndarray, grid_inner_size=[6,4], square_size:float=0.1, debug=False):
    grid_inner_size = np.array(grid_inner_size)
    chessboard_x_axis = chessboard_pose[0:3, 0:3] @ np.array([1., 0., 0.])
    chessboard_y_axis = chessboard_pose[0:3, 0:3] @ np.array([0., 1., 0.])
    topleft_corner = chessboard_pose[0:3, 3] - grid_inner_size[0]/2*square_size*chessboard_x_axis + grid_inner_size[1]/2*square_size*chessboard_y_axis
    #chessboard_corners_3d = np.zeros(((grid_inner_size[0]+1)*(grid_inner_size[1]+1), 3))
    result = []

    for x in range(grid_inner_size[0]+1):
        for y in range(grid_inner_size[1]+1):
            corner = topleft_corner + x*square_size*chessboard_x_axis - y*square_size*chessboard_y_axis
            result.append(corner)
    result = np.array(result)

    if debug:
        arrows = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)
        arrows.transform(chessboard_pose)
        corners_3D = o3d.geometry.PointCloud()
        corners_3D.points = o3d.utility.Vector3dVector(result)
        corners_3D.paint_uniform_color([1,0,0])
        o3d.visualization.draw_geometries([arrows, corners_3D])
    
    return result





def find_3D_chessboard_corners(pcd:o3d.geometry.PointCloud, matlab_engine, grid_inner_size=[6,4], square_size=0.1, debug=False):
    grid_inner_size = np.array(grid_inner_size)

    chessboard_3d_points, chessboard_intensities, chessboard_centroid, \
        chessboard_x_axis, chessboard_y_axis, chessboard_z_axis = automatic_chessboard_detection(
            pcd, matlab_engine, grid_inner_size, square_size, debug=debug
        )
    #print(get_chessboard_2d_projection_old(
    #    chessboard_3d_points, chessboard_centroid, chessboard_x_axis,
    #    chessboard_y_axis, chessboard_z_axis
    #))
    #print(get_chessboard_2d_projection(
    #    pcd, chessboard_centroid, chessboard_x_axis,
    #    chessboard_y_axis, chessboard_z_axis
    #))
    optimised_chessboard_tf = optimise_chessboard_tf(
        chessboard_3d_points, chessboard_intensities, chessboard_centroid,
        chessboard_x_axis, chessboard_y_axis, chessboard_z_axis,
        grid_inner_size, square_size, debug=debug
    )
    chessboard_pose = get_chessboard_pose(
        optimised_chessboard_tf, chessboard_centroid, chessboard_x_axis, chessboard_y_axis, chessboard_z_axis,
    )
    chessboard_corners_3d = get_chessboard_corners_3d(
        chessboard_pose, grid_inner_size, square_size, debug=False
    )

    if False:
        chessboard = o3d.geometry.OrientedBoundingBox(
            center = chessboard_pose[0:3, 3],
            R = chessboard_pose[0:3, 0:3],
            extent=[(grid_inner_size[0]+2)*square_size, (grid_inner_size[1]+2)*square_size, 0.04]
            #extent=[0.2, 1.6, 1.2]
        )
        chessboard_corners_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(chessboard_corners_3d))
        chessboard_corners_pcd.paint_uniform_color([1,0,0])
        chessboard.color = [1, 0, 0]
        chessboard_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)
        chessboard_coord_frame.transform(chessboard_pose)
        world_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)
        o3d.visualization.draw_geometries([pcd, chessboard, chessboard_corners_pcd])
    
    return chessboard_pose, chessboard_corners_3d






def find_combined_3D_chessboard_corners(
        pcds_batch: 'dict[o3d.geometry.PointCloud]',
        lidar_extrinsics: 'dict[np.ndarray]',
        matlab_engine,
        init_lidar:str = 'os128',
        grid_inner_size = [6, 4],
        square_size = 0.1,
        debug=False
):
    init_pcd = o3d.geometry.PointCloud()
    init_pcd.points = pcds_batch[init_lidar].points
    init_pcd.colors = pcds_batch[init_lidar].colors
    #init_pcd.transform(lidar_extrinsics[init_lidar])
    #o3d.visualization.draw_geometries([pcds_batch['vls128'], init_pcd])
    #init_chessboard_3d_points, chessboard_intensities, chessboard_centroid, \
    #    chessboard_x_axis, chessboard_y_axis, chessboard_z_axis = automatic_chessboard_detection(
    #        init_pcd, matlab_engine, grid_inner_size, square_size, debug=False
    #    )
    #init_chessboard_pose = get_chessboard_pose(
    #    np.zeros(5), chessboard_centroid, chessboard_x_axis, chessboard_y_axis, chessboard_z_axis
    #)
    init_chessboard_pose, _ = find_3D_chessboard_corners(
        init_pcd, matlab_engine, grid_inner_size, square_size, debug=False
    )
    print(init_chessboard_pose)
    refined_lidars_extrinsics, combined_pcd = get_combined_chessboard_3D_points(
        pcds_batch, lidar_extrinsics, init_chessboard_pose, init_lidar, grid_inner_size, square_size,
        debug=debug
    )
    return refined_lidars_extrinsics, combined_pcd

    #opt_combined_chessboard_pose = find_optimised_chessboard_3D_corners(
    #    combined_chessboard_pcd, grid_inner_size, square_size
    #)

    #if debug:
    #    cb_coord_fr = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #    cb_coord_fr.transform(opt_combined_chessboard_pose)
    #    o3d.visualization.draw_geometries([pcds_batch['vls128'], cb_coord_fr])
