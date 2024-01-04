import numpy as np
import open3d as o3d
import requests, base64
import cv2
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('agg')




def chessboard_cost_function(
    input_x:np.ndarray,
    orig_chessboard_uv:np.ndarray,
    orig_chessboard_intensities:np.ndarray,
    grid_inner_size:np.ndarray,
    square_size:float,
):
    num_squares = (grid_inner_size[0]+2) * (grid_inner_size[1]+2)

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
    return cost




def sort_corners_and_detect_chessboard_size(inp_corners:np.ndarray):
    corners = np.array(inp_corners).reshape((-1, 2))
    if corners.shape[0] == 0:
        return corners, np.array([0, 0])
    n_cols = 0
    if corners[0, 0] > corners[1, 0]:
        # If the first corner is to the right of the second corner
        # Change the order of corners
        corners = corners[::-1, :]
    for i in range(1, len(corners)):
        if corners[i, 0] < corners[i-1, 0]:
            n_cols = i
            break
    if n_cols == 0: return [], [0, 0]
    n_rows = len(corners) / n_cols
    if n_rows != int(n_rows):
        return [], [0, 0]
    else:
        n_rows = int(n_rows)
    if n_rows > n_cols:
        temp = n_cols
        n_cols = n_rows
        n_rows = temp
    if corners[0, 1] > corners[-1, 1]:
        # The first corner is below the last corner
        new_corners = np.zeros((n_rows, n_cols, 2))
        current_row = n_rows
        current_col = 0
        for i in range(len(corners)):
            current_col = i % n_cols
            if current_col == 0:
                current_row -= 1
            new_corners[current_row, current_col] = corners[i]
        corners = new_corners.reshape((-1,2))

    # At this point, the corners are from left to right, top to bottom
    # Now, we need to sort the corners from top to bottom, then left to right
    corners = corners.reshape((n_rows, n_cols, 2))
    new_corners = np.zeros((n_cols, n_rows, 2))
    for i in range(n_cols):
        new_corners[i] = corners[:, i, :]
    corners = new_corners.reshape((-1, 2))
    return corners, np.array([n_cols+1, n_rows+1])




def detect_corners_matlab(
        img:np.ndarray,
        min_corner_metric:float=0.15,
        matlab_engine=None
):
    import matlab
    import matlab.engine
    if matlab_engine is None:
        raise ValueError('Matlab engine not provided')
    if not isinstance(img, np.ndarray):
        raise ValueError('Input image must be a numpy array')
    if len(img.shape) == 3:
        temp_img = np.average(img, axis=2).astype(img.dtype)
    else:
        temp_img = img
    if img.dtype == np.uint8:
        matlab_img = matlab.uint8(temp_img.tolist())
    else:
        matlab_img = matlab.double(temp_img.tolist())
    corners, board_size = matlab_engine.detectCheckerboardPoints(
        matlab_img,
        'MinCornerMetric', min_corner_metric,
        'PartialDetections', False,
        nargout=2
    )
    corners = np.array(corners)
    if len(corners) > 0:
        board_size = np.array([board_size[0][1], board_size[0][0]], int)
    else:
        raise ValueError('No corners detected')

    # Correct the order of corners
    if corners[0,1] > corners[-1,1]:
        corners = corners[::-1, :]

    return corners, board_size



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



def redistribute_intensity(intensities:np.ndarray, ignore_zeros=False, debug=False):
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
        lower_outliers = np.where(I < min_inlier)[0]
        upper_outliers = np.where(I > max_inlier)[0]
        inliers = np.logical_and(I >= min_inlier, I <= max_inlier)
        if len(lower_outliers) == 0 and len(upper_outliers) == 0:
            break
        I[lower_outliers] = 0.
        I[upper_outliers] = 1.
        if np.max(I[inliers]) > np.min(I[inliers]):
            I[inliers] = (I[inliers] - np.min(I[inliers])) / (np.max(I[inliers]) - np.min(I[inliers]))
        break

    orig_I[valid_ids] = I
    return orig_I



def orthogonal_projection(points:np.ndarray, plane_point:np.ndarray, plane_normal:np.ndarray):
    new_plane_normal = plane_normal / np.linalg.norm(plane_normal)
    v = points - plane_point
    temp = v * new_plane_normal
    dist = temp[:,0] + temp[:,1] + temp[:,2]
    dist = dist.reshape((-1,1))
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
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    for num_iterations in range(50):
        prev_centre = bbox.center
        new_pcd = downsampled_pcd.crop(bbox)
        new_centre = np.average(np.array(new_pcd.points), axis=0)
        if np.linalg.norm(new_centre - prev_centre) <= convergent_distance:
            break
        else:
            bbox.center = prev_centre + 1.5*(new_centre - prev_centre)

    return bbox



def tf_matrix(R:np.ndarray=np.identity(3), t:np.ndarray=np.zeros((3))):
    if not isinstance(R, np.ndarray): R = np.array(R).reshape((3,3))
    if not isinstance(t, np.ndarray): t = np.array(t).reshape((3))
    result = np.identity(4)
    result[0:3, 0:3] = R
    result[0:3, 3] = t
    return result





def get_chessboard_square_id(chessboard_uv:np.ndarray, grid_inner_size=np.array([6,4], dtype=np.int8), square_size=0.1):
    grid_size = grid_inner_size + np.array([2, 2])
    chessboard_uv_ids = chessboard_uv/square_size + grid_size/2
    def sqid(uv_row):
        if uv_row[0] < 0 or uv_row[0] >= grid_size[0]: return -1
        if uv_row[1] < 0 or uv_row[1] >= grid_size[1]: return -1
        return np.trunc(uv_row[0]) + grid_size[0]*np.trunc(uv_row[1])
    chessboard_square_ids = np.apply_along_axis(sqid, 1, chessboard_uv_ids)
    chessboard_square_ids = chessboard_square_ids.astype(np.int8)
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




def get_chessboard_2d_projection(
        chessboard_3d:np.ndarray, centroid:np.ndarray, x_axis:np.ndarray, y_axis:np.ndarray, z_axis:np.ndarray,
        chessboard_tf:np.ndarray=np.zeros(3),
        grid_inner_size=[6,4], square_size=0.1
    ):

    # x_rot, y_rot, z_rot, u_trans, v_trans = chessboard_tf
    z_rot, u_trans, v_trans = chessboard_tf
    x_rot = 0.; y_rot = 0.
    chessboard_rotation = Rotation.align_vectors(
        # np.vstack((x_axis.flatten(), y_axis.flatten())), np.array([[1,0,0], [0,1,0]], np.float32)
        [x_axis, y_axis], [[1,0,0], [0,1,0]]
    )[0].as_matrix()
    rotation_adjustment = Rotation.from_euler('XYZ', [x_rot, y_rot, z_rot], degrees=False).as_matrix()
    projected_3d_chessboard = orthogonal_projection(chessboard_3d, centroid, z_axis)
    projected_3d_chessboard -= centroid
    projected_3d_chessboard = rotation_adjustment @ np.linalg.inv(chessboard_rotation) @ np.transpose(projected_3d_chessboard)
    projected_3d_chessboard = np.transpose(projected_3d_chessboard)
    #print(projected_3d_chessboard)
    chessboard_uv = projected_3d_chessboard[:, 0:2]
    chessboard_uv -= np.array([u_trans, v_trans])

    return chessboard_uv




def get_chessboard_pose(
        chessboard_tf:np.ndarray, centroid:np.ndarray, x_axis:np.ndarray, y_axis:np.ndarray, z_axis:np.ndarray,
):
    # x_rot, y_rot, z_rot, u_trans, v_trans = chessboard_tf
    z_rot, u_trans, v_trans = chessboard_tf
    x_rot = 0.; y_rot = 0.

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
