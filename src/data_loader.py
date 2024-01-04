import os, shutil, asyncio
import rosbags
from rosbags.highlevel import AnyReader
import numpy as np
import open3d as o3d
import cv2, io
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import uuid
from PIL import Image
from .utils import numpy_pc2
from scipy.spatial.transform import Rotation
from pathlib import Path
from .utils.funcs import *



class RosbagLoader:
    def __init__(self, image_topics:'list[dict]', pcd_topics:'list[dict]', temp_dir:str, data_dir:str='', pcd_accumulation_frames:int=0) -> None:
        self.image_topics = image_topics
        self.pcd_topics = pcd_topics
        self.temp_dir = temp_dir
        self.data_dir = data_dir
        self.pcd_accumulation_frames = int(pcd_accumulation_frames)

        self.cams_metadata = {}
        self.lidars_metadata = {}
        self.cam_names:'list[str]' = []
        self.lidar_names:'list[str]' = []

        for topic in image_topics:
            self.cams_metadata[topic['topic']] = topic['sensor-name']
        for topic in pcd_topics:
            self.lidars_metadata[topic['topic']] = topic['sensor-name']

        self.num_total_topics = len(self.image_topics) + len(self.pcd_topics)

        # Automatic chessboard detection parameters
        self.collected_centers = []
        self.collected_rvecs = []



    def load_image_from_msg(self, msg):
        if msg.__msgtype__.find('CompressedImage') > -1:
            img_arr = np.fromstring(msg.data, np.uint8)
            img_arr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        elif msg.__msgtype__.find('Image') > -1:
            height = int(msg.height)
            width = int(msg.width)
            img_arr = msg.data.reshape((height, width, -1))
            if img_arr.shape[2] == 1: img_arr = img_arr.reshape((height, width))
        else:
            raise RuntimeError
        img = Image.fromarray(img_arr)
        img_dir = os.path.join(self.temp_dir, f'{uuid.uuid1()}.png')
        img.save(img_dir)
        return img_dir



    def load_pcd_from_msg(self, msg, lidar_name:str=''):
        def get_init_tf(lidar_name:str):
            if lidar_name == 'pandar64':
                init_tf = np.identity(4)
                init_tf[0:3, 0:3] = Rotation.from_euler('z', [90], degrees=True).as_matrix()
            elif lidar_name == 'falcon':
                init_tf = np.identity(4)
                init_tf[0:3, 0:3] = Rotation.from_euler('yx', [90, 180], degrees=True).as_matrix()
            else:
                init_tf = np.identity(4)
            return init_tf

        pcd_raw = numpy_pc2.pointcloud2_to_array(msg)
        pcd_data = pcd_raw[['x', 'y', 'z', 'intensity']]
        pcd = np.transpose(np.vstack((
            pcd_data['x'].astype(np.float32).flatten(),
            pcd_data['y'].astype(np.float32).flatten(),
            pcd_data['z'].astype(np.float32).flatten(),
            pcd_data['intensity'].astype(np.float32).flatten()
        )))
        pcd = np.transpose(get_init_tf(lidar_name) @ np.transpose(pcd))
        return pcd


    async def _single_rosbag_loader(
            self, bag_file:str,
            target_times:'list[float]',
            topics:'list[str]'=[],
            timesync_criteria:float=0.1,
            queue:asyncio.Queue=None,
            callback=None,
    ):
        first_loop = True
        cam_names = []
        lidar_names = []

        if not isinstance(topics, list): topics = [topics]
        if bag_file.endswith('.db3'):
            validated_bag_file = os.path.dirname(bag_file)
        elif bag_file.endswith('.bag'):
            validated_bag_file = bag_file
        else:
            raise ValueError('Invalid bag file extension')

        bag_reader = AnyReader([Path(validated_bag_file)])
        bag_reader.open()
        connections = []

        if len(topics) > 0:
            for connection in bag_reader.connections:
                if connection.topic in topics:
                    connections.append(connection)
            for topic in topics:
                try:
                    cam_names.append(self.cams_metadata[topic])
                except: pass
                try:
                    lidar_names.append(self.lidars_metadata[topic])
                except: pass
        else:
            cam_names:'list[str]' = list(self.cams_metadata.values())
            lidar_names:'list[str]' = list(self.lidars_metadata.values())
        sensor_names = cam_names + lidar_names
        # print('Loading data for sensors:', sensor_names)

        image_data = {tt: {cam_name: None for cam_name in cam_names} for tt in target_times}
        pcd_data = {tt: {lidar_name: None for lidar_name in lidar_names} for tt in target_times}

        current_target_time_index = 0
        num_topics = len(topics)
        if num_topics == 0: num_topics = self.num_total_topics

        msgs = {}
        sensor_timestamps:'dict[str, list[float]]' = {sensor_name: [] for sensor_name in sensor_names}

        def find_closest_timestamps(sensor_timestamps):
            all_sync_sets = []
            sets_widths = []
            current_pivot_time = np.max([sensor_timestamps[sensor_name][0] for sensor_name in sensor_names])
            start_time = np.min([sensor_timestamps[sensor_name][0] for sensor_name in sensor_names])

            done = False
            while True:
                current_set = {}
                next_sensor_times = {}
                # print('Current pivot time:', current_pivot_time)
                for sensor_name in sensor_names:
                    current_sensor_time_id = np.where(sensor_timestamps[sensor_name] <= current_pivot_time)[0][-1]
                    current_set[sensor_name] = sensor_timestamps[sensor_name][current_sensor_time_id]
                    try:
                        next_sensor_times[sensor_name] = sensor_timestamps[sensor_name][current_sensor_time_id+1]
                    except IndexError:
                        done = True
                        break
                if done: break
                # print('Current sensors set:', current_set)
                current_times = list(current_set.values())
                set_width = np.max(current_times) - np.min(current_times)
                # print('Current set width:', set_width)
                all_sync_sets.append(current_set)
                sets_widths.append(set_width)
                current_pivot_time = np.max(list(next_sensor_times.values()))
                if current_pivot_time - start_time > 1.:
                    break

            min_timesync_id = np.argmin(sets_widths)
            min_timesync = sets_widths[min_timesync_id] / 1e9
            if min_timesync > timesync_criteria:
                # print(f'Timesync criteria is not satisfied: {min_timesync}s > {timesync_criteria}s')
                raise ValueError('Timesync criteria is not satisfied')
            
            return all_sync_sets[min_timesync_id]


        pbar_tqdm = tqdm(total=len(target_times)-1, desc='Reading & processing rosbag data')
        for connection, t_nanosec, rawdata in bag_reader.messages(connections=connections):
            topic = connection.topic
            t = t_nanosec / 1e9
            if first_loop:
                first_loop = False
                t0 = t

            if 0 <= t - t0 - target_times[current_target_time_index]:
                # Collect timestamps of all sensors
                try:
                    cam_name = self.cams_metadata[topic]
                except: cam_name = None
                try:
                    lidar_name = self.lidars_metadata[topic]
                except: lidar_name = None
                if cam_name is None and lidar_name is None: continue

                sensor_name = cam_name if cam_name is not None else lidar_name
                msg = bag_reader.deserialize(rawdata, connection.msgtype)
                msgs[str(t_nanosec)] = msg
                sensor_timestamps[sensor_name].append(t_nanosec)

            if t - t0 - target_times[current_target_time_index] > 2.:
                # Timestamp sync
                try:
                    synced_sensor_timestamps = find_closest_timestamps(sensor_timestamps)
                    for sensor, timestamp in synced_sensor_timestamps.items():
                        if sensor in cam_names:
                            current_img = self.load_image_from_msg(msgs[str(timestamp)])
                            current_cam_name = sensor
                            image_data[target_times[current_target_time_index]][sensor] = current_img
                        elif sensor in lidar_names:
                            current_lidar_name = sensor
                            # If accumulation is enabled, find adjacent timestamps to accumulate point clouds
                            if self.pcd_accumulation_frames > 0:
                                all_pcd_timestamps = np.array(sensor_timestamps[sensor])
                                timestamp_id = int(np.where(all_pcd_timestamps == timestamp)[0][0])
                                start_id = max(0, timestamp_id-self.pcd_accumulation_frames)
                                end_id = min(len(all_pcd_timestamps)-1, timestamp_id+self.pcd_accumulation_frames)
                                pcd_timestamps = all_pcd_timestamps[start_id : end_id+1]
                            else:
                                pcd_timestamps = [timestamp]
                            current_pcd = None
                            for pcd_timestamp in pcd_timestamps:
                                if not isinstance(current_pcd, np.ndarray):
                                    current_pcd = self.load_pcd_from_msg(msgs[str(pcd_timestamp)], sensor)
                                else:
                                    pcd_data[target_times[current_target_time_index]][sensor] = np.vstack((
                                        current_pcd,
                                        self.load_pcd_from_msg(msgs[str(pcd_timestamp)], sensor),
                                    ))
                            pcd_data[target_times[current_target_time_index]][sensor] = current_pcd
                        # processed_sensors.append(sensor)
                    try:
                        callback(current_img, current_cam_name, current_pcd, current_lidar_name)
                    except: pass
                except: pass
                current_target_time_index += 1
                if current_target_time_index >= len(target_times): break
                msgs = {}
                sensor_timestamps:'dict[str, list[float]]' = {sensor_name: [] for sensor_name in sensor_names}
                if queue is not None:
                    await queue.put(current_target_time_index/(len(target_times)-1))
                pbar_tqdm.update(1)
                # processed_sensors = []

        if len(self.cam_names) == 0:
            self.cam_names:'list[str]' = cam_names
        if len(self.lidar_names) == 0:
            self.lidar_names:'list[str]' = lidar_names

        bag_reader.close()
        pbar_tqdm.close()
        return image_data, pcd_data



    def load_from_single_rosbag_file(self, bag_file:str, target_times:'list[float]', topics:'list[str]'=[], save_dir=''):
        print('Loading from single rosbag at times:', target_times)
        target_times = sorted(target_times)
        async def _worker():
            image_data, pcd_data = self._single_rosbag_loader(
                bag_file = bag_file,
                target_times = target_times,
                topics = topics
            )
            return image_data, pcd_data
        task = asyncio.create_task(_worker())
        image_data, pcd_data = asyncio.run(task)

        print('Loading done')

        if save_dir != '':
            try:
                shutil.rmtree(os.path.join(save_dir, 'calib', self.cam_names[0]))
            except: pass

            # Save images from one image topic
            os.makedirs(os.path.join(save_dir, 'calib', self.cam_names[0], 'imgs'), exist_ok=True)
            for i, target_time in enumerate(target_times):
                temp_img_file = image_data[target_time][self.cam_names[0]]
                shutil.copyfile(temp_img_file, os.path.join(save_dir, 'calib', self.cam_names[0], 'imgs', f'{i}.png'))

            for lidar_name in self.lidar_names:
                os.makedirs(os.path.join(save_dir, 'calib', self.cam_names[0], lidar_name), exist_ok=True)
                for i, target_time in enumerate(target_times):
                    pcd = pcd_data[target_time][lidar_name]
                    np.save(os.path.join(save_dir, 'calib', self.cam_names[0], lidar_name, f'{i}.npy'), pcd)




    def load_from_multiple_rosbag_files(self, bag_files:'list[str]',  target_time=2., save_dir=''):
        all_image_data = []
        all_pcd_data = []

        async def _worker(bag_file, target_time):
            image_data, pcd_data = self._single_rosbag_loader(
                bag_file = bag_file,
                target_times = [target_time],
                topics = []
            )
            return image_data, pcd_data

        for i, bag_file in enumerate(bag_files):
            task = asyncio.create_task(_worker(bag_file, target_time))
            image_data, pcd_data = asyncio.run(task)
            all_image_data.append(image_data)
            all_pcd_data.append(pcd_data)

        if save_dir != '':
            cam_name = self.cam_names[0]
            try:
                shutil.rmtree(os.path.join(save_dir, 'calib', cam_name))
            except: pass

            # Save images from one image topic
            os.makedirs(os.path.join(save_dir, 'calib', cam_name, 'imgs'), exist_ok=True)
            for i, image_data in enumerate(all_image_data):
                temp_img_file = image_data[target_time][cam_name]
                shutil.copyfile(temp_img_file, os.path.join(save_dir, 'calib', cam_name, 'imgs', f'{i}.png'))

            for lidar_name in self.lidar_names:
                os.makedirs(os.path.join(save_dir, 'calib', cam_name, lidar_name), exist_ok=True)
                for i, pcd_data in enumerate(all_pcd_data):
                    pcd = pcd_data[target_time][lidar_name]
                    np.save(os.path.join(save_dir, 'calib', cam_name, lidar_name, f'{i}.npy'), pcd)
    




    def enhance_lidar_image(self, intensity_img:np.ndarray, densify=True, kernel_size:int=3):
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



    def hybrid_3D_chessboard_detector(
            self,
            inp_img:np.ndarray,
            cam_name:str,
            inp_pcd:np.ndarray,
            lidar_name:str,
            debug=False,
    ):
        try:
            self.matlab_engine
        except:
            import matlab
            import matlab.engine
            self.matlab_engine = matlab.engine.start_matlab()

        chessboard_detected = False
        I = inp_pcd[:, 3]
        I /= np.max(I)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inp_pcd[:, 0:3])
        pcd.colors = o3d.utility.Vector3dVector(np.vstack((I, I, I)).T)

        pcd_down = pcd.voxel_down_sample(voxel_size=0.005)
        pcd_down.estimate_normals()
        grid_inner_size = self.grid_inner_size
        square_size = self.chessboard_square_size
        planar_patches = pcd_down.detect_planar_patches(
            coplanarity_deg = 0.9,
            min_plane_edge_length=grid_inner_size[0]*square_size,
        )
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
                img_arr = (plt.imread(buf) / 255.)
                img_arr = img_arr[:,:,0].astype(np.float32)
                img_arr = self.enhance_lidar_image(img_arr, densify=True, kernel_size=2)

            corners, board_size = detect_corners_matlab(img_arr, 0.2, self.matlab_engine)
            if len(corners) > 0 and board_size[0]*board_size[1] >= 8:
                est_chessboard_bbox = new_bbox
                chessboard_detected = True
                break

        if debug and chessboard_detected:
            plt.imsave('debug/rosbag_hybrid_chessboard_detection.png', img_arr, cmap='gray')

        try:
            if self.is_good_sample(est_chessboard_bbox, force_ignore=True):
                count = len(self.collected_centers)
                print(f'Collected {count} samples', flush=True)
                img_dir = os.path.join(self.save_dir, 'calib', cam_name, 'imgs', f'{count-1}.png')
                # print(f'Saving image to {img_dir}')
                shutil.copyfile(inp_img, img_dir)
                pcd_dir = os.path.join(self.save_dir, 'calib', cam_name, lidar_name, f'{count-1}.npy')
                # print(f'Saving point cloud to {pcd_dir}')
                np.save(pcd_dir, inp_pcd)
        except:
            return None



    async def auto_poses_finder(
            self, bag_file:str, 
            image_topic:str, pcd_topic:str, 
            ncols:int, nrows:int, square_size:float,
            save_dir:str,
            start_time:float=10., end_time:float=None, time_interval:float=1.,
            queue:asyncio.Queue=None,
    ):
        # Load necessary pair of images and point clouds
        if end_time is None:
            if bag_file.endswith('.db3'):
                bag_dir = os.path.dirname(bag_file)
            elif bag_file.endswith('.bag'):
                bag_dir = bag_file
            with AnyReader([Path(bag_dir)]) as bag_reader:
                end_time = bag_reader.duration/1e9 - 10.
        
        target_times = np.arange(start_time, end_time, time_interval)
        cam_name = self.cams_metadata[image_topic]
        lidar_name = self.lidars_metadata[pcd_topic]
        self.grid_inner_size = np.array([ncols-2, nrows-2])
        self.chessboard_square_size = square_size
        self.save_dir = save_dir
        self.collected_centers = []
        self.collected_rvecs = []

        try:
            shutil.rmtree(os.path.join(save_dir, 'calib', cam_name))
        except: pass
        os.makedirs(os.path.join(save_dir, 'calib', cam_name, 'imgs'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'calib', cam_name, lidar_name), exist_ok=True)

        image_data, pcd_data = await self._single_rosbag_loader(
            bag_file = bag_file,
            target_times = target_times,
            topics = [image_topic, pcd_topic],
            timesync_criteria = 0.05, # Stricter timesync criteria
            queue = queue,
            callback = self.hybrid_3D_chessboard_detector,
        )




    
    def is_good_sample(self, inp_bbox:o3d.geometry.OrientedBoundingBox, force_ignore=False):
        if inp_bbox is None:
            return False
        
        rvec = Rotation.from_matrix(np.array(inp_bbox.R)).as_euler('xyz', degrees=True)
        centre = np.array(inp_bbox.center)

        if len(self.collected_centers) == 0:
            self.collected_centers.append(centre.tolist())
            self.collected_rvecs.append(rvec.tolist())
            # print(self.collected_centers)
            # print(self.collected_rvecs)
            return True
        
        collected_centers = np.array(self.collected_centers)
        collected_rvecs = np.array(self.collected_rvecs)
        
        dists = np.linalg.norm(collected_centers - centre, axis=1)
        min_dist_id = np.argmin(dists)
        min_dist = dists[min_dist_id]
        if min_dist < 0.05 and not force_ignore:
            return False
        elif min_dist < 0.3 and not force_ignore:
            # If current pose is in proximity of one of the already collected poses,
            # check rotation variability
            rot_diff = np.linalg.norm(collected_rvecs[min_dist_id] - rvec)
            if rot_diff < 10.:
                return False

        self.collected_centers.append(inp_bbox.center.tolist())
        self.collected_rvecs.append(rvec.tolist())
        # print(self.collected_centers)
        # print(self.collected_rvecs)
        return True




def sort_list_of_dicts_by_key(dicts:'list[dict]', key):
    sorted_ids = np.argsort([d[key] for d in dicts])
    result:'list[dict]' = [dicts[i] for i in sorted_ids]
    return result
