import os
from glob import glob
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Text3D
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


#_matlab_engine_startup = matlab.engine.start_matlab(background=True)
#matlab_engine = _matlab_engine_startup.result()





def tf_matrix(R:np.ndarray=np.identity(3), t:np.ndarray=np.zeros((3))):
    if not isinstance(R, np.ndarray): R = np.array(R).reshape((3,3))
    if not isinstance(t, np.ndarray): t = np.array(t).reshape((3))
    result = np.identity(4)
    result[0:3, 0:3] = R
    result[0:3, 3] = t.flatten()
    return result





def calib_camera_intrinsics(data_dir:str, camera_name:str, selected_poses:list, matlab_engine, square_size=0.1, force_recalib=False, debug=False):
    json_dir = os.path.join(data_dir, camera_name, '{}_params.json'.format(camera_name))
    #json_dir = os.path.join('./calib_data', 'cams', '{}_params.json'.format(camera_name))
    if os.path.exists(json_dir) and not force_recalib:
        print('Found intrinsics parameters for', camera_name)
        with open(json_dir, 'r') as json_file:
            result = json.loads(json_file.read())

    else:
        #extra_image_file_names = glob(os.path.join('./20230327/rgb-thermal-calib/rgb_0', '*.png'))
        #image_file_names = []
        cam_prefix = 'rgb' if camera_name.find('rgb') > -1 else 't'
        image_file_names = sorted(glob('./20230720/calib_data/{}_0/imgs/*.jpg'.format(cam_prefix)))
        #for j in range(1):
            #for i in [0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23]:
        #    for i in range(24):
        #        image_file_names.append(os.path.join('./20230520/pos{}/calib_data'.format(j), 'cams', '{}_{}'.format(cam_prefix, j), '{}.jpg'.format(i)))
        #for ff in extra_image_file_names:
        #    image_file_names.append(ff)
        print('Calibrating {} using {} images'.format(camera_name, len(image_file_names)))
        #print(image_file_names)
        img0 = np.array(Image.open(image_file_names[0]), np.uint8)
        detector = matlab_engine.vision.calibration.monocular.CheckerboardDetector()
        print('Detecting chessboard patterns')
        imagePoints, imagesUsed = matlab_engine.detectPatternPoints(detector, image_file_names, nargout=2)
        #imagePoints, boardSize = matlab_engine.detectCheckerboardPoints(image_file_names, nargout=2)
        #worldPoints = matlab_engine.generateCheckerboardPoints(boardSize, matlab.double(100))
        if camera_name.find('rgb') > -1:
            num_radial_distort_coefs = 3
            estimate_skew = True
            estimate_tangent = False
        elif camera_name.find('t') > -1:
            num_radial_distort_coefs = 3
            estimate_skew = True
            estimate_tangent = False
        worldPoints = matlab_engine.generateWorldPoints(detector, 'SquareSize', square_size*1000)
        #imagesUsed = np.array(imagesUsed)[0]
        #if np.any(imagesUsed == False):
        #    print('Could not detect chessboard corners in images: {}'.format(
        #        np.where(imagesUsed == False)[0]
        #    ))

        print('Calculating camera parameters')
        cameraParams, imagesUsed = matlab_engine.estimateCameraParameters(
            imagePoints, worldPoints,
            'WorldUnits', 'millimeters',
            'NumRadialDistortionCoefficients', num_radial_distort_coefs,
            'EstimateSkew', estimate_skew,
            'EstimateTangentialDistortion', estimate_tangent,
            'ImageSize', matlab.double([img0.shape[1], img0.shape[0]]),
            nargout=2
        )

        result = {
            'CameraName': camera_name,
            'IntrinsicMatrix': np.array(matlab_engine.getfield(cameraParams, 'K')).tolist(),
            'RadialDistortion': np.array(matlab_engine.getfield(cameraParams, 'RadialDistortion')).tolist(),
            'TangentialDistortion': np.array(matlab_engine.getfield(cameraParams, 'TangentialDistortion')).tolist(),
            'Skew': float(matlab_engine.getfield(cameraParams, 'Skew')),
            'ImageSize': np.array(matlab_engine.getfield(cameraParams, 'ImageSize'), int)[0].tolist(),
        }
        #if result['IntrinsicMatrix'][1][2] < 0: result['IntrinsicMatrix'][1][2] = -result['IntrinsicMatrix'][1][2]
        with open(json_dir, 'w') as json_file:
            json_file.write(json.dumps(result, indent=4))
    
    if debug:
        print(result)
        img_file = os.path.join(data_dir, camera_name, 'imgs', '5.jpg')
        img = np.array(Image.open(img_file), np.uint8)
        undistorted_img = undistort_image(img, result, debug=True)
    return result
        
        




def undistort_image_old(image:np.ndarray, cam_params:dict, matlab_engine, debug=False):
    if debug:
        fig, axs = plt.subplots(1, 2, figsize=(15,10))
        axs[0].axis('off'); axs[1].axis('off')
        axs[0].imshow(image)
        axs[0].set_title('Original image')

    M = np.array(cam_params['IntrinsicMatrix'])
    f = M[[0,1], [0,1]]
    c = M[[0,1], [2,2]]

    #ml_cam_params = matlab_engine.cameraIntrinsics(
    #    'K', matlab.double(np.transpose(cam_params['IntrinsicMatrix']).tolist()),
    #    'RadialDistortion', matlab.double(cam_params['RadialDistortion']),
    #    'TangentialDistortion', matlab.double(cam_params['TangentialDistortion']),
    #    'Skew', matlab.double(cam_params['Skew'])
    #)
    ml_cam_params = matlab_engine.cameraIntrinsics(
        matlab.double(f.tolist()),
        matlab.double(c.tolist()),
        matlab.double([cam_params['ImageSize'][1], cam_params['ImageSize'][0]]),
        'RadialDistortion', matlab.double(cam_params['RadialDistortion']),
        'TangentialDistortion', matlab.double(cam_params['TangentialDistortion']),
        'Skew', matlab.double(cam_params['Skew'])
    )
    undistorted_img_matlab = matlab_engine.undistortImage(matlab.uint8(image.tolist()), ml_cam_params)
    undistorted_img = np.array(undistorted_img_matlab, dtype=np.uint8)
    
    if debug:
        axs[1].imshow(undistorted_img)
        axs[1].set_title('Undistorted image')
    return undistorted_img




def undistort_image(image:np.ndarray, cam_params:dict, debug=False):
    if debug:
        fig, axs = plt.subplots(1, 2, figsize=(15,10))
        axs[0].axis('off'); axs[1].axis('off')
        axs[0].imshow(image)
        axs[0].set_title('Original image')
    
    K = np.array(cam_params['IntrinsicMatrix'])
    k1, k2, k3 = cam_params['RadialDistortion'][0]
    p1, p2 = cam_params['TangentialDistortion'][0]
    w, h = cam_params['ImageSize']
    dist = np.array([k1, k2, p1, p2, k3])

    #new_K, roi = cv.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    undst_img = cv.undistort(image, K, dist, None, None)
    #x, y, w, h = roi
    #undst_img = undst_img[y:y+h, x:x+w]
    if debug:
        axs[1].imshow(undst_img)
        axs[1].set_title('Undistorted image')
    return undst_img




def get_retina_corrected_image(image:np.ndarray, debug=False):
    retina = cv.bioinspired_Retina.create((image.shape[1], image.shape[0]))
    #retina.write('retinaParams.xml')
    retina.setup('retinaParams.xml')
    retina.run(image)
    retina_parvo = retina.getParvo()
    retina_parvo = np.array(retina_parvo, dtype=np.uint8)
    if debug:
        fig, axs = plt.subplots(1, 2, figsize=(15,10))
        axs[0].imshow(image)
        axs[0].set_title('Original image')
        axs[1].imshow(retina_parvo)
        axs[1].set_title('Bio-retina enhanced image')
    return retina_parvo




def detect_image_corners(inp_image:np.ndarray, cam_params:dict, matlab_engine, downscale=1, debug=False):
    ml_cam_params = matlab_engine.cameraParameters(
        'IntrinsicMatrix', matlab.double(np.transpose(cam_params['IntrinsicMatrix']).tolist()),
        'RadialDistortion', matlab.double(cam_params['RadialDistortion']),
        'TangentialDistortion', matlab.double(cam_params['TangentialDistortion']),
    )
    #undistorted_img_ml = matlab.uint8(undistorted_img.tolist())
    #img = get_retina_corrected_image(img)
    if len(inp_image.shape) == 3:
        img = np.average(inp_image, axis=2).astype(np.uint8)
    else: img = inp_image
    if downscale != 1:
        img = np.array(cv.resize(img, [img.shape[1]//downscale, img.shape[0]//downscale]))
        #plt.imshow(img)
    undistorted_img = undistort_image(inp_image, cam_params)
    ml_img = matlab.uint8(img.tolist())
    
    if debug:
        fig, axs = plt.subplots(1, 2, figsize=(15,10))
        undst_inp_img = undistort_image(inp_image, cam_params)
        axs[0].imshow(inp_image, cmap='gray')
        axs[0].set_title('Input image')
        axs[1].imshow(undistorted_img, cmap='gray')
        axs[1].set_title('Corners detection result')

    raw_img_corners, board_size = matlab_engine.detectCheckerboardPoints(
        ml_img, 
        'MinCornerMetric', 0.1, 
        'PartialDetections', False, 
        nargout=2
    )
    raw_img_corners = np.array(raw_img_corners, dtype=np.float32)
    if downscale != 1:
        raw_img_corners *= downscale
    if debug:
        disp_img_corners = np.array(raw_img_corners, dtype=np.float32)
        axs[0].scatter(disp_img_corners[1:-1,0], disp_img_corners[1:-1,1], s=10, facecolors='none', edgecolors='r')
        axs[0].scatter(disp_img_corners[0,0], disp_img_corners[0,1], s=10, facecolors='none', edgecolors='y', marker='s')
        axs[0].scatter(disp_img_corners[-1,0], disp_img_corners[-1,1], s=10, facecolors='none', edgecolors='g', marker='s')

    if 1:
        img_corners = matlab_engine.undistortPoints(raw_img_corners, ml_cam_params)
    if 0:
        K = np.array(cam_params['IntrinsicMatrix'])
        k1, k2, k3 = cam_params['RadialDistortion'][0]
        p1, p2 = cam_params['TangentialDistortion'][0]
        w, h = cam_params['ImageSize']
        dist = np.array([k1, k2, p1, p2, k3])
        img_corners = cv.undistortPoints(raw_img_corners, K, dist)

    img_corners = np.array(img_corners, dtype=np.float32).reshape((-1,2))
    #print(img_corners)
    board_size = np.array(board_size, dtype=int)

    if debug:
        disp_img_corners = np.array(img_corners, dtype=np.float32)
        axs[1].scatter(disp_img_corners[1:-1,0], disp_img_corners[1:-1,1], s=10, facecolors='none', edgecolors='r')
        axs[1].scatter(disp_img_corners[0,0], disp_img_corners[0,1], s=10, facecolors='none', edgecolors='y', marker='s')
        axs[1].scatter(disp_img_corners[-1,0], disp_img_corners[-1,1], s=10, facecolors='none', edgecolors='g', marker='s')

    return img_corners, undistorted_img, board_size




def detect_image_corners_v2(img:np.ndarray, cam_params:dict, matlab_engine, downscale=1, debug=False):
    undistorted_img = undistort_image(img, cam_params)
    #undistorted_img_ml = matlab.uint8(undistorted_img.tolist())
    #img = get_retina_corrected_image(img)
    if len(img.shape) == 3:
        img = np.average(img, axis=2).astype(np.uint8)
    if downscale != 1:
        img = np.array(cv.resize(img, [img.shape[1]//downscale, img.shape[0]//downscale]))
        #plt.imshow(img)
    ml_img = matlab.uint8(img.tolist())
    
    if debug:
        fig, axs = plt.subplots(1, 2, figsize=(15,10))
        axs[0].imshow(ml_img, cmap='gray')
        axs[0].set_title('Input image')
        axs[1].imshow(undistorted_img, cmap='gray')
        axs[1].set_title('Corners detection result')

    raw_img_corners, board_size = matlab_engine.detectCheckerboardPoints(
        ml_img, 
        'MinCornerMetric', 0.12, 
        'PartialDetections', False, 
        nargout=2
    )
    img_corners = matlab_engine.undistortPoints(raw_img_corners, ml_cam_params)

    img_corners = np.array(img_corners, dtype=np.float32)
    board_size = np.array(board_size, dtype=int)
    if downscale != 1:
        img_corners *= downscale

    if debug:
        disp_img_corners = np.array(img_corners, dtype=np.float32)
        axs[1].scatter(disp_img_corners[1:-1,0], disp_img_corners[1:-1,1], s=10, facecolors='none', edgecolors='r')
        axs[1].scatter(disp_img_corners[0,0], disp_img_corners[0,1], s=10, facecolors='none', edgecolors='y', marker='s')
        axs[1].scatter(disp_img_corners[-1,0], disp_img_corners[-1,1], s=10, facecolors='none', edgecolors='g', marker='s')

    return img_corners, undistorted_img, board_size




def initial_camera_pose_estimate(world_points_3D:np.ndarray, img_corners:np.ndarray, cam_params:dict, matlab_engine):
    world_corners_ml = matlab.double(world_points_3D.tolist())
    img_corners_ml = matlab.double(img_corners.tolist())

    # Process the camera intrinsics
    cam_intrinsics = matlab_engine.cameraIntrinsics(
        # [fx, fy]
        matlab.double([cam_params['IntrinsicMatrix'][0][0], cam_params['IntrinsicMatrix'][1][1]]),
        # [cx, cy]
        matlab.double([cam_params['IntrinsicMatrix'][0][2], cam_params['IntrinsicMatrix'][1][2]]),
        matlab.double(cam_params['ImageSize']),
        'RadialDistortion', matlab.double(cam_params['RadialDistortion']),
        'TangentialDistortion', matlab.double(cam_params['TangentialDistortion']),
        'Skew', matlab.double(cam_params['Skew'])
    )
    #cam_intrinsics = matlab_engine.cameraParameters(
    #    'IntrinsicMatrix', matlab.double(np.transpose(cam_params['IntrinsicMatrix']).tolist()),
    #    'RadialDistortion', matlab.double(cam_params['RadialDistortion']),
    #    'TangentialDistortion', matlab.double(cam_params['TangentialDistortion']),
    #)

    #print('Estimating camera pose by solving PnP 2D-3D correspondences problem')
    cam_pose_raw, inlierIdx = matlab_engine.estworldpose(
        img_corners_ml, world_corners_ml, cam_intrinsics,
        'MaxReprojectionError', 20.0,
        'MaxNumTrials', 5000,
        nargout=2
    )
    inlierIdx = np.array(inlierIdx)
    print(len(inlierIdx))
    cam_pose = np.array(matlab_engine.getfield(cam_pose_raw, 'A'))
    #print('Estimated camera pose:\n', cam_pose)

    #corners = np.array(img_corners)
    #fig, ax = plt.subplots(figsize=(10,10))
    #ax.imshow(np.array(undistorted_img_ml))
    #ax.scatter(corners[:,0], corners[:,1], s=20, facecolors='none', edgecolors='r')
    #ax.set_title(t.to_sec())

    return cam_pose





def estimate_camera_pose_P3P(
        corners_in_all_images_list:'list[np.ndarray]', 
        corners_in_all_pcds_list:'list[np.ndarray]', 
        cam_params:dict, 
        matlab_engine
):
    corners_in_all_images = np.vstack(corners_in_all_images_list)
    corners_in_all_pcds = np.vstack(corners_in_all_pcds_list)
    
    cam_pose = initial_camera_pose_estimate(corners_in_all_pcds, corners_in_all_images, cam_params, matlab_engine)
    return cam_pose







def initial_camera_pose_estimate_v2(world_points_3D:np.ndarray, img_corners:np.ndarray, cam_params:dict, matlab_engine):
    M = cam_params['IntrinsicMatrix']
    fx = M[0][0]
    fy = M[1][1]
    cx = M[0][2]
    cy = M[1][2]
    #print(world_points_3D.shape, img_corners.shape)
    R, T, R_GN, T_GN = CPnP(world_points_3D.T, img_corners.T, fx, fy, cx, cy)

    cam_pose = tf_matrix(R, T)
    return cam_pose







def PnP_reprojection_error(cam_pose:np.ndarray, world_corners_3D:np.ndarray, img_corners:np.ndarray, cam_intrinsic_matrix:np.ndarray, debug=False, debug_image:np.ndarray=None):
    lidar_to_cam = np.linalg.inv(cam_pose)
    cam_intrinsic_matrix = np.array(cam_intrinsic_matrix)
    #cam_intrinsic_matrix[0,1] = 0.
    world_points_3D = np.hstack((world_corners_3D, np.ones((world_corners_3D.shape[0], 1))))
    
    cam_points_3D = np.transpose(lidar_to_cam @ np.transpose(world_points_3D))
    cam_points_2D = np.transpose(cam_intrinsic_matrix @ np.transpose(cam_points_3D[:,0:3]))
    cam_points_2D /= cam_points_2D[:,2].reshape((-1,1))
    #print(cam_points_2D); print(img_corners)

    reprojection_error = np.sqrt( (cam_points_2D[:,0]-img_corners[:,0])**2 + (cam_points_2D[:,1]-img_corners[:,1])**2 )
    reprojection_error = np.sum(reprojection_error) / img_corners.shape[0]

    if debug:
        cam_pose = np.array(cam_pose)
        board_distance = np.linalg.norm(cam_pose[:2, 3] - np.average(world_corners_3D, axis=0)[:2])
        fig, ax = plt.subplots(figsize=(10,10))
        ax.axis('off')
        if len(debug_image.shape) == 2:
            ax.imshow(debug_image, cmap='gray')
        else:
            debug_image = get_retina_corrected_image(debug_image)
            ax.imshow(debug_image)
        ax.scatter(cam_points_2D[:,0], cam_points_2D[:,1], s=10, marker='x', c='r', label='2D re-projection of 3D corners')
        ax.scatter(img_corners[:,0], img_corners[:,1], s=30, marker='o', facecolors='none', edgecolors='g', label='Corners detected in 2D image')
        #ax.scatter(img_corners[:,0], img_corners[:,1], s=10, marker='x', c='r', label='Corners detected in 2D image')
        ax.set_title("Target's distance = {:.2f}m, 3D–2D re-projection error = {:.2f}".format(board_distance, reprojection_error))
        ax.legend()

    return reprojection_error





def plot_reprojection_error(cam_pose:np.ndarray, valid_ids:list, all_world_corners_3D:'list[np.ndarray]', all_imgs_corners:'list[np.ndarray]', cam_params:dict):
    rp_errors = []
    for i in range(len(all_world_corners_3D)):
        rp_errors.append(PnP_reprojection_error(
            cam_pose, all_world_corners_3D[i], all_imgs_corners[i], cam_params['IntrinsicMatrix']
        ))
    fig, ax = plt.subplots(figsize=(10,5))
    print(rp_errors)
    ax.bar(list(range(len(rp_errors))), rp_errors, tick_label=valid_ids)
    ax.grid()
    ax.set_xlabel('Pose index')
    ax.set_ylabel('3D – 2D re-projection error [pixels]')
    ax.set_title('Mean re-projection error = {:.2f}'.format(np.average(rp_errors)))
    plt.show()





def PnP_reprojection_error_in_3D_old(
        cam_pose:np.ndarray, img_corners:np.ndarray, world_corners:np.ndarray, cam_intrinsics, matlab_engine
):
    img_corners_ml = matlab.double(img_corners.tolist())
    world_corners_ml = matlab.double(world_corners.tolist())
    #reprojected_cam_extrinsic = matlab_engine.estimateExtrinsics(image_points, world_points, cam_intrinsics)
    reprojected_cam_extrinsic, _ = matlab_engine.estworldpose(
        img_corners_ml, world_corners_ml, cam_intrinsics,
        'MaxReprojectionError', 1.,
        'MaxNumTrials', 5000,
        nargout=2
    )
    reprojected_cam_pose = np.array(matlab_engine.getfield(reprojected_cam_extrinsic, 'A'), np.float32)

    orig_rot_angles = Rotation.from_matrix(cam_pose[:3, :3]).as_euler('xyz', True)
    reprojected_rot_angles = Rotation.from_matrix(reprojected_cam_pose[:3, :3]).as_euler('xyz', True)
    #print(orig_rot_angles, reprojected_rot_angles)
    rot_error = np.linalg.norm(orig_rot_angles - reprojected_rot_angles)
    trans_error = np.linalg.norm(cam_pose[:3, 3] - reprojected_cam_pose[:3, 3])
    print(cam_pose[:3,3], reprojected_cam_pose[:3,3])
    print(rot_error, trans_error)
    return rot_error, trans_error


def PnP_reprojection_error_in_3D(
        cam_pose:np.ndarray, img_corners:np.ndarray, world_corners_3D, cam_intrinsic_matrix:np.ndarray
):
    # Determine scale factor w
    num_corners = img_corners.shape[0]
    world_corners_3D_homo = np.vstack((world_corners_3D.T, np.ones(num_corners)))
    world_corners_2D_projected = cam_intrinsic_matrix @ (np.linalg.inv(cam_pose) @ world_corners_3D_homo)[:3, :]
    w = world_corners_2D_projected[2, :]

    # Reproject things back
    img_corners_homo = np.vstack((img_corners.T, np.ones(num_corners)))
    temp_img_corners_3d = np.linalg.inv(cam_intrinsic_matrix) @ img_corners_homo
    temp_img_corners_3d = np.vstack((temp_img_corners_3d, np.ones(temp_img_corners_3d.shape[1])))
    
    reprojected_img_corners_3d = (cam_pose @ temp_img_corners_3d).T
    reprojected_img_corners_3d = reprojected_img_corners_3d[:,:3] * w.reshape((-1,1))

    #print(reprojected_img_corners_3d); print(world_corners_3D)
    #print(np.linalg.norm(reprojected_img_corners_3d - world_corners_3D, axis=1))
    #raise RuntimeError
    #print(world_corners_3D / reprojected_img_corners_3d)

    reprojection_error = np.average(np.linalg.norm(reprojected_img_corners_3d - world_corners_3D, axis=1))
    return reprojection_error




def plot_PnP_reprojection_error_3D(
        cam_pose:np.ndarray, chessboard_poses:'list[np.ndarray]', all_img_corners:'list[np.ndarray]',
        all_3D_corners:'list[np.ndarray]', cam_params
):
    #world_points = matlab_engine.generateCheckerboardPoints(matlab.uint8([8, 6]), 0.1)
    cam_pose = np.array(cam_pose)
    
    reprojection_errors = []; board_distances = []
    for i in range(len(all_img_corners)):
        #reprojection_error = PnP_reprojection_error_in_3D(
        #    cam_pose, all_img_corners[i], all_3D_corners[i], cam_params['IntrinsicMatrix']
        #)
        reprojection_error = PnP_reprojection_error(
            cam_pose, all_3D_corners[i], all_img_corners[i], cam_params['IntrinsicMatrix']
        )
        reprojection_errors.append(reprojection_error)
        distance = np.linalg.norm(cam_pose[:2,3] - chessboard_poses[i][:2,3])
        board_distances.append('{:.2f}'.format(distance))
    
    print('avg:', np.average(reprojection_errors))
    print([f'{err:.2f}' for err in reprojection_errors])

    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    ax.bar(np.arange(len(all_img_corners)), reprojection_errors, tick_label=board_distances)
    ax.grid()
    ax.set_xlabel('Distance from camera [m]')
    ax.set_ylabel('3D – 2D re-projection error [pixels]')






def estimate_camera_pose(image:np.ndarray, cam_params:np.ndarray, world_corners_3D:np.ndarray, matlab_engine, debug=False, pose_debug=None):
    undistorted_img = undistort_image(image, cam_params, matlab_engine, debug=False)
    #retina_img = get_retina_corrected_image(image, debug=debug)
    #retina_img = undistorted_img
    img_corners, board_size = detect_image_corners(image, cam_params, matlab_engine, debug=debug)

    # If the first corner is not the top-left corner,
    # i.e., the first corner is to the right of the last corner
    if img_corners[0,0] > img_corners[-1,0]:
        img_corners = img_corners[::-1, :]
    
    if pose_debug in [0, 2, 3, 7]:
        img_corners = img_corners.reshape((-1, 5, 2))
        img_corners = img_corners[1:8, :].reshape((-1, 2))
    elif pose_debug in [9]:
        img_corners = img_corners.reshape((-1, 5, 2))
        img_corners = img_corners[0:7, :].reshape((-1, 2))
    elif pose_debug in [1]:
        img_corners = img_corners.reshape((-1, 5, 2))
        img_corners = img_corners[2:9, :].reshape((-1, 2))

    if debug:
        print(len(img_corners))
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(undistorted_img)
        ax.scatter(img_corners[1:-1,0], img_corners[1:-1,1], s=10, facecolors='none', edgecolors='r')
        ax.scatter(img_corners[0,0], img_corners[0,1], s=10, facecolors='none', edgecolors='y', marker='s')
        ax.scatter(img_corners[-1,0], img_corners[-1,1], s=10, facecolors='none', edgecolors='g', marker='s')
    cam_pose = initial_camera_pose_estimate(world_corners_3D, img_corners, cam_params, matlab_engine)
    return cam_pose, img_corners, undistorted_img





def optimise_camera_pose(initial_cam_poses:'list[np.ndarray]', all_world_corners_3D:'list[np.ndarray]', all_imgs_corners:'list[np.ndarray]', cam_params:dict):
    num_poses = len(all_world_corners_3D)
    initial_cam_pose = initial_cam_poses[0]
    if len(initial_cam_poses) > 1:
        for i in range(1, len(initial_cam_poses)):
            initial_cam_pose += initial_cam_poses[i]
        initial_cam_pose /= num_poses
    cam_intrinsic_matrix = np.array(cam_params['IntrinsicMatrix'])
    
    current_cam_params = {}
    for key in cam_params.keys():
        current_cam_params[key] = cam_params[key]

    def overall_reprojection_error(pose_adjustment:np.ndarray, return_cam_pose=False):
        '''pose_adjustment =[
            x_rot, y_rot, z_rot, x_trans, y_trans, z_trans,
            delta_fx, delta_fy, delta_cx, delta_cy,
        ]
        '''
        rotation_adjustment = Rotation.from_euler('XYZ', pose_adjustment[0:3], degrees=True).as_matrix()
        #translation_adjustment = pose_adjustment[3:6]
        #tf_adjustment = tf_matrix(rotation_adjustment, translation_adjustment)
        #cam_pose = tf_adjustment @ initial_cam_pose
        cam_pose = initial_cam_pose.copy()
        cam_pose[0:3, 3] += pose_adjustment[3:6]
        cam_pose[0:3, 0:3] = rotation_adjustment @ initial_cam_pose[0:3, 0:3]
        #current_cam_intrinsic_matrix = cam_intrinsic_matrix.copy()
        #current_cam_intrinsic_matrix[[0,1,0,1], [0,1,2,2]] += pose_adjustment[6:10]
        overall_error = 0
        for i in range(num_poses):
            overall_error += PnP_reprojection_error(
                cam_pose, all_world_corners_3D[i], all_imgs_corners[i], cam_intrinsic_matrix,
            )
        #overall_error /= num_poses
        if return_cam_pose:
            return cam_pose
        else:
            return overall_error


    upper_bounds = np.array([30, 30, 30, 2, 2, 2], dtype=np.float32)
    
    opt_result = scipy.optimize.minimize(
        fun = overall_reprojection_error,
        x0 = np.zeros(6),
        method = 'Powell',
        bounds = scipy.optimize.Bounds(-upper_bounds, upper_bounds),
        options = {
            "ftol": 1e-12
        }
    )
    print(opt_result)

    return overall_reprojection_error(opt_result.x, return_cam_pose=True)





def sensor_fusion(inp_image:np.ndarray, pcd_o3d:o3d.geometry.PointCloud, cam_pose:np.ndarray, cam_params:np.ndarray, retina_correction=False):
    cam_pose = np.array(cam_pose)
    if retina_correction:
        image = get_retina_corrected_image(inp_image)
    else:
        image = inp_image
    lidar_to_cam =  np.linalg.inv(cam_pose)
    #pcd_no_hidden_points, _ = pcd_o3d.hidden_point_removal(
    #    camera_location = cam_pose[:3, 3],
    #    radius = 1000000
    #)
    #pcd = np.array(pcd_no_hidden_points.vertices)
    pcd = np.array(pcd_o3d.points)
    pcd_points = np.hstack((pcd, np.ones((pcd.shape[0], 1))))
    cam_3d_points = np.transpose(lidar_to_cam @ np.transpose(pcd_points))
    cam_intrinsic_matrix = np.array(cam_params['IntrinsicMatrix'])
    #cam_intrinsic_matrix[0, 1] = 0.
    #image = undistort_image(image, cam_params, matlab_engine)

    cam_2d_points = np.transpose(cam_intrinsic_matrix @ np.transpose(cam_3d_points[:, 0:3]))
    valid_w = np.array(cam_2d_points[:,2] >= 0)
    cam_2d_points /= cam_2d_points[:,2].reshape((-1,1))
    cam_2d_points = np.round(cam_2d_points).astype(int)

    image_size = cam_params['ImageSize']
    valid_u = np.logical_and(0 <= cam_2d_points[:,0], cam_2d_points[:,0] < image_size[0])
    valid_v = np.logical_and(0 <= cam_2d_points[:,1], cam_2d_points[:,1] < image_size[1])
    valid_uv = np.logical_and(valid_u, valid_v)
    valid_2d_points = np.where(np.logical_and(valid_uv, valid_w) == True)[0]
    
    colours = np.zeros((pcd.shape))
    if len(image.shape) == 2: #Thermal image
        pixel_colours = np.zeros((image.shape[0], image.shape[1], 3))
        for i in range(3):
            pixel_colours[:,:,i] = image/255.0
    elif len(image.shape) == 3:
        pixel_colours = image/255.0

    for valid_index in valid_2d_points:
        colours[valid_index, :] = pixel_colours[cam_2d_points[valid_index][1], cam_2d_points[valid_index][0], :]

    fused_pcd = o3d.geometry.PointCloud()
    #fused_pcd.points = pcd_no_hidden_points.vertices
    fused_pcd.points = o3d.utility.Vector3dVector(pcd)
    fused_pcd.colors = o3d.utility.Vector3dVector(colours)

    return fused_pcd






def visualise_cameras_and_chessboard_poses(cam_poses:'list[np.ndarray]', chessboard_pose:np.ndarray, pcd:o3d.geometry.PointCloud, grid_inner_size=[6,4], square_size=0.1):
    grid_inner_size = np.array(grid_inner_size)
    chessboard = o3d.geometry.OrientedBoundingBox(
        center = chessboard_pose[0:3, 3],
        R = chessboard_pose[0:3, 0:3],
        extent=[(grid_inner_size[0]+2)*square_size, (grid_inner_size[1]+2)*square_size, 0.05]
        #extent=[0.2, 1.6, 1.2]
    )
    chessboard.color = [1, 0, 0]
    chessboard_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)
    chessboard_coord_frame.transform(chessboard_pose)
    world_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)

    geometries = [chessboard, chessboard_coord_frame, pcd]
    for i in range(len(cam_poses)):
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)
        #cam_pose = np.linalg.inv(cam_poses[i])
        coord.transform(cam_poses[i])
        geometries.append(coord)
    geometries.append(world_coord_frame)
    
    o3d.visualization.draw_geometries(geometries)








def project_lidar_on_image(
        inp_image:np.ndarray, pcd_o3d:o3d.geometry.PointCloud, cam_pose:np.ndarray, cam_params:np.ndarray,
        depth_max=35.0, point_size=2.
):
    #lidar_to_cam = np.linalg.inv(cam_pose) @ np.linalg.inv(cb_origin_world_to_cam) @ lidar_pose
    #lidar_to_cam = tf_matrix(R=Rotation.from_euler('yz', [-90, 90], degrees=True).as_matrix())
    if cam_params['CameraName'].find('rgb') > -1:
        image = get_retina_corrected_image(inp_image)
    else:
        image = inp_image
    cam_intrinsic_matrix = np.array(cam_params['IntrinsicMatrix'])
    #cam_intrinsic_matrix[0, 1] = 0.
    
    pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd_o3d)

    rgbd_img = pcd_t.project_to_rgbd_image(
        width=cam_params['ImageSize'][0],
        height=cam_params['ImageSize'][1],
        intrinsics=cam_intrinsic_matrix,
        extrinsics=np.linalg.inv(cam_pose),
        depth_max=depth_max, depth_scale=1.0
    )

    #intensity_img = np.asarray(rgbd_img.color, np.float32)
    depth_img = np.asarray(rgbd_img.depth, np.float32)
    #print(np.where(depth_img != 0))
    valid_ids = np.where(depth_img != 0)
    
    fig, ax = plt.subplots(figsize=(20,20))
    ax.axis('off')
    if cam_params['CameraName'].find('rgb') > -1:
        ax.imshow(image)
        #plt.imsave('./demo/cams/rgb_0/0_rectified.jpg', image)
    else:
        ax.imshow(image, cmap='gray')
    ax.scatter(valid_ids[1], valid_ids[0], c=depth_img[valid_ids[0], valid_ids[1]], s=point_size, cmap='nipy_spectral')



    return






def Zhou_calibration(
        images: 'list[np.ndarray]',
        chessboard_poses: 'list[np.ndarray]',
        all_pcds_o3d: 'list[o3d.geometry.PointCloud]',
        valid_ids: list,
        cam_params: dict,
        matlab_engine,
        square_size = 0.1
):
    cam_intrinsics = matlab_engine.cameraIntrinsics(
        # [fx, fy]
        matlab.double([cam_params['IntrinsicMatrix'][0][0], cam_params['IntrinsicMatrix'][1][1]]),
        # [cx, cy]
        matlab.double([cam_params['IntrinsicMatrix'][0][2], cam_params['IntrinsicMatrix'][1][2]]),
        matlab.double(cam_params['ImageSize']),
        'RadialDistortion', matlab.double(cam_params['RadialDistortion']),
        'TangentialDistortion', matlab.double(cam_params['TangentialDistortion']),
        'Skew', matlab.double(cam_params['Skew'])
    )
    num_poses = len(chessboard_poses)
    #all_img_corners_3D = np.zeros((4, 3, 1))
    #all_img_corners_3D = np.zeros((4, 3, 1))
    #all_chessboard_pcds = []
    matlab_engine.eval('chessboard_pcds = {};', nargout=0)
    count = -1
    for i in valid_ids:
        count += 1
        if len(images[i].shape) == 3:
            img = np.average(images[i], axis=2).astype(np.uint8)
        else: img = images[i]
        img_ml = matlab.uint8(img.tolist())
        img_corners_3D = np.array(matlab_engine.estimateCheckerboardCorners3d(
            img_ml, cam_intrinsics, square_size*1000,
            'MinCornerMetric', 0.1,
            'Padding', matlab.double([160.5, 40.5, 160.5, 40.5])
        ))
        #print(img_corners_3D)
        if img_corners_3D.shape[0] == 0: continue
        img_corners_3D = img_corners_3D.reshape((4, 3, 1))
        try:
            all_img_corners_3D = np.append(all_img_corners_3D, img_corners_3D, axis=2)
        except:
            all_img_corners_3D = img_corners_3D
        #print(all_img_corners_3D)
        #print(np.array(img_corners_3D))
        #raise RuntimeError
        #all_img_corners_3D.append(img_corners_3D)
        chessboard_pose = chessboard_poses[i]
        chessboard_bbox = o3d.geometry.OrientedBoundingBox(
            center = chessboard_pose[:3,3],
            R = chessboard_pose[:3,:3],
            #extent = [0.8, 0.6, 0.1]
            extent = [1.2, 0.7, 0.1]
        )
        chessboard_pcd_o3d = all_pcds_o3d[i].crop(chessboard_bbox)
        matlab_engine.workspace['chessboard_pcd'] = matlab_engine.pointCloud(
            matlab.double(np.array(chessboard_pcd_o3d.points).tolist())
        )
        #all_chessboard_pcds.append(chessboard_pcd)
        matlab_engine.eval('chessboard_pcds = [chessboard_pcds, chessboard_pcd];', nargout=0)
    
    print('Estimating')
    all_img_corners_3D = matlab.double(all_img_corners_3D.tolist())
    all_chessboard_pcds = matlab_engine.workspace['chessboard_pcds']
    tf_ml = matlab_engine.estimateLidarCameraTransform(all_chessboard_pcds, all_img_corners_3D)
    tf = np.array(matlab_engine.getfield(tf_ml, 'A'))
    return np.linalg.inv(tf)















def CPnP(s, Psens_2D, fx, fy, u0, v0):
    from scipy.linalg import expm, eigh, eig, svd
    N = s.shape[1]
    bar_s = np.mean(s, axis=1).reshape(3,1)
    Psens_2D = Psens_2D - np.array([[u0],[v0]])
    obs = Psens_2D.reshape((-1, 1), order="F")
    pesi = np.zeros((2*N,11))
    G = np.ones((2*N,1))
    W = np.diag([fx, fy])
    M = np.hstack([np.kron(bar_s.T, np.ones((2*N,1))) - np.kron(s.T, np.ones((2,1))), np.zeros((2*N, 8))])

    for k in range(N):
        pesi[[2*k],:] = np.hstack([-(s[0,k]-bar_s[0]) * obs[2*k], -(s[1,k]-bar_s[1]) * obs[2*k], -(s[2,k]-bar_s[2]) * obs[2*k], (fx * s[:,[k]]).T.tolist()[0], fx, 0, 0, 0, 0])
        pesi[[2*k+1],:] = np.hstack([-(s[0,k]-bar_s[0]) * obs[2*k+1], -(s[1,k]-bar_s[1]) * obs[2*k+1], -(s[2,k]-bar_s[2]) * obs[2*k+1], 0, 0, 0, 0, (fy * s[:,[k]]).T.tolist()[0], fy])

    J = np.dot(np.vstack([pesi.T, obs.T]), np.hstack([pesi, obs])) / (2*N)
    delta= np.vstack([np.hstack( [np.dot(M.T, M), np.dot(M.T, G)]), np.hstack( [np.dot(G.T,M), np.dot(G.T, G)] )]) / (2*N)

    w, D = eig(J, delta)    
    sigma_est = min(abs(w))

    est_bias_eli= np.dot(np.linalg.inv((np.dot(pesi.T,pesi)- sigma_est*(np.dot(M.T, M)))/(2*N)) , (np.dot(pesi.T,obs) - sigma_est * np.dot(M.T, G))/ (2*N) )
    bias_eli_rotation = np.vstack([est_bias_eli[3:6].T, est_bias_eli[7:10].T, est_bias_eli[0:3].T])
    bias_eli_t= np.hstack([est_bias_eli[6],  est_bias_eli[10], 1-bar_s[0]*est_bias_eli[0]-bar_s[1]*est_bias_eli[1]-bar_s[2]*est_bias_eli[2]]).T
    normalize_factor= np.linalg.det(bias_eli_rotation) ** (1/3)
    bias_eli_rotation=bias_eli_rotation / normalize_factor
    t = bias_eli_t/normalize_factor

    U, x, V = svd(bias_eli_rotation)
    V = V.T

    RR = np.dot(U, np.diag([1, 1, np.linalg.det(np.dot(U, V.T))]))
    R = np.dot(RR, V.T)

    E = np.array([[1, 0, 0],[0, 1, 0]])
    WE = np.dot(W, E)
    e3 = np.array([[0],[0],[1]])
    J = np.zeros((2*N, 6))

    g = np.dot(WE, np.dot(R, s) + np.tile(t,N).reshape(N,3).T)
    h = np.dot(e3.T, np.dot(R, s) + np.tile(t,N).reshape(N,3).T)

    f = g/h
    f =  f.reshape((-1, 1), order="F")
    I3 = np.diag([1, 1, 1])

    for k in range(N):
        J[[2*k, 2*k+1],:] = np.dot((WE * h[0, k] - g[:,[k]]* e3.T), np.hstack([s[1,k]*R[:,[2]]-s[2,k]*R[:,[1]], s[2,k]*R[:,[0]]-s[0,k]*R[:,[2]], s[0,k]*R[:,[1]]-s[1,k]*R[:,[0]], I3])) / h[0, k]**2

    initial = np.hstack([np.zeros((3)), t.tolist()]).reshape(6,1)
    results = initial + np.dot(np.dot(np.linalg.inv(np.dot(J.T, J)), J.T), (obs - f))
    X_GN = results[0:3]
    t_GN = results[3:6]
    Xhat = np.array([
        [0, -X_GN[2], X_GN[1]], 
        [X_GN[2], 0, -X_GN[0]],
        [-X_GN[1], X_GN[0], 0]
    ])
    R_GN = np.dot(R, expm(Xhat))

    return R,t,R_GN,t_GN






def UPnP(corners_in_all_images_list:'list[np.ndarray]', corners_in_all_pcds_list:'list[np.ndarray]', cam_params:dict):
    corners_in_all_images = np.vstack(corners_in_all_images_list)
    corners_in_all_pcds = np.vstack(corners_in_all_pcds_list)
    print(corners_in_all_images.shape, corners_in_all_pcds.shape)

    W, H = cam_params['ImageSize']
    def pixel2angle(pix):
        u = pix[0]
        v = pix[1]
        inclination = (H / 2 - u) * np.pi / H
        azimuth = - (v - W / 2) * 2 * np.pi / W
        return [inclination, azimuth]
    
    img_bearing_vectors = np.zeros((corners_in_all_images.shape[0], 3))
    #for i, pix in enumerate(corners_in_all_images):
    #    angs = pixel2angle(pix)
    #    img_bearing_vectors[i] = [np.cos(angs[0]) * np.cos(angs[1]), np.cos(angs[0]) * np.sin(angs[1]), np.sin(angs[0])]
    tmp_corners_in_img = np.hstack([corners_in_all_images[:, [0,1]], 1 + np.zeros(corners_in_all_images.shape[0]).reshape(-1, 1)])
    print(tmp_corners_in_img.shape)
    inv_K = np.linalg.inv(cam_params['IntrinsicMatrix'])
    for i, pix in enumerate(tmp_corners_in_img):
        tmp = np.dot(inv_K, pix.T).T
        img_bearing_vectors[i] = tmp / np.linalg.norm(tmp)
    
    pcd_bearing_vectors = np.array(corners_in_all_pcds) / np.linalg.norm(corners_in_all_pcds, axis=1).reshape(-1, 1)

    transformation = pyopengv.absolute_pose_ransac(img_bearing_vectors, pcd_bearing_vectors, "UPNP", 0.001, 100000)
    transformation = tf_matrix(transformation[:3, :3], transformation[:3, 3])
    #transformation = np.linalg.inv(transformation)
    return transformation



def estimate_camera_pose_UPnP(images:'list[np.ndarray]', corners_in_all_pcds_list:'list[np.ndarray]', cam_params:dict, matlab_engine):
    corners_in_all_images_list = []
    for img in images:
        img_corners, _ = detect_image_corners(img, cam_params, matlab_engine)
        corners_in_all_images_list.append(img_corners)
    
    transformation = UPnP(corners_in_all_images_list, corners_in_all_pcds_list, cam_params)
    print(transformation)
    return transformation






def visualise_all_chessboard_poses_o3d(pose_ids:list, all_chessboard_poses:'list[np.ndarray]', cam_pose:np.ndarray=None):
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    ground_plane = o3d.geometry.TriangleMesh.create_box(100, 50, 0.05)
    ground_plane.translate([-7, -25, -1.5])
    ground_plane.paint_uniform_color([0.5, 0.5, 0.5])
    geometries = [coord_frame, ground_plane]
    if isinstance(cam_pose, list):
        cam_pose = np.array(cam_pose)
    if isinstance(cam_pose, np.ndarray):
        cam_coord = o3d.geometry.TriangleMesh(coord_frame)
        cam_coord.transform(cam_pose)
        geometries.append(cam_coord)

    num_chessboards = len(all_chessboard_poses)
    for i in range(num_chessboards):
        if 1:
            chessboard = o3d.geometry.TriangleMesh.create_box(0.8, 0.6, 0.05)
            chessboard.transform(all_chessboard_poses[i])
            chessboard.paint_uniform_color(plt.get_cmap('gist_ncar')(i/num_chessboards)[:3])
        else:
            chessboard = o3d.geometry.OrientedBoundingBox(
                center = all_chessboard_poses[i][:3,3],
                R = all_chessboard_poses[i][:3,:3],
                extent = [0.8, 0.6, 0.05]
            )
            chessboard.color = plt.get_cmap('gist_ncar')(i/num_chessboards)[:3]
        geometries.append(chessboard)
    
    o3d.visualization.draw_geometries(geometries)





def visualise_all_chessboard_poses(pose_ids:list, all_chessboard_poses:'list[np.ndarray]', lidar_name:str, cam_pose:np.ndarray=None):
    geometries = []
    if isinstance(cam_pose, list):
        cam_pose = np.array(cam_pose)
    num_chessboards = len(all_chessboard_poses)

    fig = plt.figure(figsize=(10,7))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    # Draw LiDAR coordinate frame
    x_ax = ax.quiver(0, 0, 0, 1, 0, 0, length=2, arrow_length_ratio=0.1)
    y_ax = ax.quiver(0, 0, 0, 0, 1, 0, arrow_length_ratio=0.1)
    z_ax = ax.quiver(0, 0, 0, 0, 0, 1, arrow_length_ratio=0.1)
    x_ax.set(color='r')
    y_ax.set(color='g')
    z_ax.set(color='b')
    #ax.add_collection3d(Text3D(0, 0, 0, 'Ouster OS1-128 \n LiDAR sensor'))
    ax.text(0, 0, 0, f'      {lidar_name} \n LiDAR origin')

    for i in range(num_chessboards):
        chessboard = o3d.geometry.TriangleMesh.create_box(0.8, 0.6, 0.05)
        chessboard.transform(all_chessboard_poses[i])
        chessboard.paint_uniform_color(plt.get_cmap('gist_ncar')(i/num_chessboards)[:3])
        
        verts = np.array(chessboard.vertices)[[0, 1, 5, 4], :]
        cb = Poly3DCollection([verts.tolist()])
        cb.set_facecolor(plt.get_cmap('gist_ncar')(i/num_chessboards)[:3])
        ax.add_collection3d(cb)

        if 0:
            xy_projection = verts.copy()
            xy_projection[:, 2] = -2.
            xy_proj = Poly3DCollection([xy_projection.tolist()])
            xy_proj.set_facecolor(plt.get_cmap('gist_ncar')(i/num_chessboards)[:3])
            ax.add_collection3d(xy_proj)

    ax.set_xlim(0, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-2, 6)
    ax.view_init(elev=10, azim=190, roll=0)
    plt.show()