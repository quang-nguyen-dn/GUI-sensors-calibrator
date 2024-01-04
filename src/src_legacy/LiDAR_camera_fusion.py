import numpy as np
import open3d as o3d
import cv2
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from glob import glob
import json
from PIL import Image
import matlab
import matlab.engine



def get_retina_corrected_image(image:np.ndarray, debug=False):
    retina = cv2.bioinspired_Retina.create((image.shape[1], image.shape[0]))
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



def undistort_image(image:np.ndarray, cam_params:dict, matlab_engine, debug=False):
    if debug:
        fig, axs = plt.subplots(1, 2, figsize=(15,10))
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



def perform_sensor_fusion(inp_image:np.ndarray, pcd_o3d:o3d.geometry.PointCloud, cam_params:dict, retina_correction=False):
    cam_pose = np.array(cam_params['Extrinsic'])
    if retina_correction:
        image = get_retina_corrected_image(inp_image)
    else:
        image = inp_image
    lidar_to_cam =  np.linalg.inv(cam_pose)
    #pcd_no_hidden_points, non_hidden_points_ids = pcd_o3d.hidden_point_removal(
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
    
    #colours = np.zeros((pcd.shape))
    colours = np.array(pcd_o3d.colors)
    if len(image.shape) == 2: #Thermal image
        pixel_colours = np.zeros((image.shape[0], image.shape[1], 3))
        for i in range(3):
            pixel_colours[:,:,i] = image/255.0
    elif len(image.shape) == 3:
        pixel_colours = image/255.0

    for valid_index in valid_2d_points:
        if np.any(colours[valid_index] != 0): continue
        try:
            colours[valid_index, :] = pixel_colours[cam_2d_points[valid_index][1], cam_2d_points[valid_index][0], :]
        except: pass

    fused_pcd = o3d.geometry.PointCloud()
    fused_pcd.points = o3d.utility.Vector3dVector(pcd)
    fused_pcd.colors = o3d.utility.Vector3dVector(colours)

    return fused_pcd



class LiDAR_camera_fusion:
    def __init__(self) -> None:
        self.CAM_NAMES = ['rgb_{}'.format(i) for i in range(5)]
        self.LIDAR_NAMES = ['vls128', 'os128', 'os64', 'pandar64', 'livox', 'falcon']
        self.CAM_PARAMS = {cam_name: None for cam_name in self.CAM_NAMES}
        self.images = {cam_name: None for cam_name in self.CAM_NAMES}
        self.labels = {cam_name: None for cam_name in self.CAM_NAMES}

    def load_data(self, DATA_DIR, matlab_engine):
        self.DATA_DIR = DATA_DIR
        for cam_name in self.CAM_NAMES:
            cam_params_path = os.path.join('/data2/sensors-calib/20230520/calib_data/', cam_name, '{}_params.json'.format(cam_name))
            with open(cam_params_path, 'r') as cam_json_file:
                self.CAM_PARAMS[cam_name] = json.loads(cam_json_file.read())

            img_path = os.path.join(DATA_DIR, 'cams', cam_name, '0.jpg')
            img = Image.open(img_path)
            img = np.array(img)
            #retina_img = get_retina_corrected_image(img)
            #plt.imsave(os.path.join(DATA_DIR, 'cams', cam_name, '0_retina.jpg'), retina_img)
            undistorted_img = undistort_image(img, self.CAM_PARAMS[cam_name], matlab_engine)
            self.images[cam_name] = undistorted_img
            
            try:
                labels_img = Image.open(
                    os.path.join(DATA_DIR, 'cams/LabelingProject/GroundTruthProject/PixelLabelData/', 'Label_{}_0_retina.png'.format(int(cam_name[-1])+1))
                )
                labels_img = np.array(labels_img, np.uint8)
                labels_image = plt.get_cmap('tab10')(labels_img-1)[:,:,:3]
                labels_image[labels_img==0] = 0
                labels_image = (labels_image*255.0).astype(np.uint8)
                labels_image = undistort_image(labels_image, self.CAM_PARAMS[cam_name], matlab_engine)
                self.labels[cam_name] = labels_image
                #plt.imshow(undistorted_img); plt.show()
            except:
                pass
        print('Finished loading data')

    def sensor_fusion(self, in_pcd:o3d.geometry.PointCloud, retina_correction=False):
        fused_pcd = o3d.geometry.PointCloud()
        fused_pcd.points = in_pcd.points
        fused_pcd.paint_uniform_color([0,0,0])
        for cam_name in self.CAM_NAMES:
            temp_fused_pcd = perform_sensor_fusion(self.images[cam_name], fused_pcd, self.CAM_PARAMS[cam_name], retina_correction)
            #o3d.visualization.draw_geometries([temp_fused_pcd])
            fused_pcd.colors = temp_fused_pcd.colors
        return fused_pcd
    
    def label_fusion(self, in_pcd:o3d.geometry.PointCloud):
        labeled_pcd = o3d.geometry.PointCloud(in_pcd)
        labeled_pcd.paint_uniform_color([0,0,0])
        for cam_name in self.labels.keys():
            if not isinstance(self.labels[cam_name], np.ndarray): continue
            print('Fusing labels from camera', cam_name)
            temp_labeled_pcd = perform_sensor_fusion(self.labels[cam_name], labeled_pcd, self.CAM_PARAMS[cam_name])
            labeled_pcd.colors = temp_labeled_pcd.colors
        return labeled_pcd



if __name__ == '__main__':
    _matlab_engine_startup = matlab.engine.start_matlab(background=True)
    pcd = np.load('/data3/material-segmentation/fusion_points/0/lidars/os128/0.npy')
    pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd[:, :3]))
    #pcd_o3d = o3d.io.read_point_cloud('../fusion_points/combined_map.pcd')
    fuser = LiDAR_camera_fusion()
    matlab_engine = _matlab_engine_startup.result()
    fuser.load_data('/data3/material-segmentation/fusion_points/0/', matlab_engine)
    fused_pcd = fuser.sensor_fusion(pcd_o3d, True)
    #o3d.io.write_point_cloud('../fusion_points/fused_combined_map.pcd', fused_pcd)
    o3d.visualization.draw_geometries([fused_pcd])

    #labeled_pcd = fuser.label_fusion(pcd_o3d)
    #o3d.io.write_point_cloud('../fusion_points/labeled_combined_map.pcd', labeled_pcd)
