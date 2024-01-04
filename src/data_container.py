import os
import numpy as np
import re
from glob import glob
from PIL import Image

class DataContainer:
    def __init__(self, cam_metadata:'dict', lidars_metadata:'list[dict]', data_dir:str, selected_poses:'list[int]'=None) -> None:
        self.cam_metadata = cam_metadata
        self.lidars_metadata = lidars_metadata
        self.data_dir = data_dir
        if selected_poses is None or len(selected_poses) == 0:
            selected_poses = np.arange(cam_metadata['data-count'])
        self.selected_poses = selected_poses

        self.cam_name:str = cam_metadata['sensor-name']
        self.lidar_names:'list[str]' = [lidar_data['sensor-name'] for lidar_data in lidars_metadata]

        self.images:'list[np.ndarray]' = []
        self.point_clouds:'dict[str, list[np.ndarray]]' = {lidar_name: [] for lidar_name in self.lidar_names}

        self.load_images()
        self.load_pcds()

        self.valid_ids = np.arange(len(self.images), dtype=int)


    def load_images(self):
        self.images = []
        img_paths = sorted(glob(os.path.join(self.data_dir, 'calib', self.cam_name, 'imgs/*.png')))
        img_nums = []
        for img_path in img_paths:
            img_num = re.search('[0-9][0-9]\.png', img_path)
            if img_num is not None:
                img_num = int(img_num.group()[0:2])
            else:
                img_num = re.search('[0-9]\.png', img_path)
                if img_num is None:
                    raise ValueError('Please check the naming of image files in the data directory')
                else:
                    img_num = int(img_num.group()[0])
            img_nums.append(img_num)
        img_paths = [img_paths[i] for i in np.argsort(img_nums)]
        for img_path in img_paths:
            image = np.array(Image.open(img_path))
            if self.cam_metadata['colour'] == 'RGB':
                image = image[:, :, 0:3]
            elif self.cam_metadata['colour'] == 'Gray':
                image = image[:, :]
            self.images.append(image)
        print(f'Loaded {len(self.images)} images')


    def load_pcds(self):
        for lidar_name in self.lidar_names:
            self.point_clouds[lidar_name] = []
            pcd_files = sorted(glob(os.path.join(self.data_dir, 'calib', self.cam_name, lidar_name, '*.npy')))
            pcd_nums = []
            for pcd_file in pcd_files:
                pcd_num = re.search('[0-9][0-9]\.npy', pcd_file)
                if pcd_num is not None:
                    pcd_num = int(pcd_num.group()[0:2])
                else:
                    pcd_num = re.search('[0-9]\.npy', pcd_file)
                    if pcd_num is None:
                        raise ValueError('Please check the naming of point cloud files in the data directory')
                    else:
                        pcd_num = int(pcd_num.group()[0])
                pcd_nums.append(pcd_num)
            pcd_files = [pcd_files[i] for i in np.argsort(pcd_nums)]
            for pcd_file in pcd_files:
                pcd_arr = np.load(pcd_file)
                # pcd_arr[:, 2] -= 1.
                self.point_clouds[lidar_name].append(pcd_arr)
            print(f'Loaded {len(self.point_clouds[lidar_name])} point clouds for {lidar_name}')
