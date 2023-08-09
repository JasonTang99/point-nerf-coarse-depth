import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import open3d as o3d
import matplotlib.pyplot as plt
import copy

# Gets the intrinsics of depth and color streams
def get_intrinsics(depth_ints_path, color_ints_path, 
        bag_fp = "data/aligned/color_sample.bag"):
    
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(
        config, 
        bag_fp,
        repeat_playback=False
    )

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)

    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    depth_ints = {
        'coeffs': depth_intrinsics.coeffs,
        'fx': depth_intrinsics.fx,
        'fy': depth_intrinsics.fy,
        'height': depth_intrinsics.height,
        'ppx': depth_intrinsics.ppx,
        'ppy': depth_intrinsics.ppy,
        'width': depth_intrinsics.width
    }

    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    color_ints = {
        'coeffs': color_intrinsics.coeffs,
        'fx': color_intrinsics.fx,
        'fy': color_intrinsics.fy,
        'height': color_intrinsics.height,
        'ppx': color_intrinsics.ppx,
        'ppy': color_intrinsics.ppy,
        'width': color_intrinsics.width
    }

    # Save Intrinsics to file
    with open(depth_ints_path, 'w') as f:
        json.dump(depth_ints, f)
    with open(color_ints_path, 'w') as f:
        json.dump(color_ints, f)

# intrinsics json to o3d intrinsics
def _read_intrinsics(path):
    with open(path, 'r') as f:
        ints = json.load(f)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = ints["width"],
            height = ints["height"],
            fx = ints["fx"],
            fy = ints["fy"],
            cx = ints["ppx"],
            cy = ints["ppy"]
        )
    return intrinsics

# Return o3d wrapped intrinsics
def get_o3d_intrinsics():
    depth_path = "data/intrinsics/depth_intrinsics.json"
    color_path = "data/intrinsics/color_intrinsics.json"
    depth_intrinsics = _read_intrinsics(depth_path)
    color_intrinsics = _read_intrinsics(color_path)
    return depth_intrinsics, color_intrinsics


# COLMAP Script from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_dense.py
def read_colmap_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1,
            usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


########### O3D UTILS ###########
def colordepth_to_rgbd(color_img, depth_img, depth_scale=1000.0):
    assert len(color_img.shape) == 3
    assert len(depth_img.shape) == 2 or \
        (len(depth_img.shape) == 3 and depth_img.shape[2] == 1)
    assert color_img.shape[0] == depth_img.shape[0] and \
        color_img.shape[1] == depth_img.shape[1]
    
    color_raw = o3d.geometry.Image(color_img)
    depth_raw = o3d.geometry.Image(depth_img)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False, 
        depth_trunc=1000, depth_scale=depth_scale)
    return rgbd_image

def rgbd_to_pcd(rgbd_image, intrinsics, extrinsics=np.eye(4), voxel_size=None):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        intrinsics, 
        extrinsics
    )
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd

def gen_pcds(color_imgs, depth_imgs, intrinsics, cb_scale=1.0, depth_scale=1000.0):
    pcds_full = []
    for color_img, depth_img in zip(color_imgs, depth_imgs):
        # Color + Depth -> RGBD image -> Pointcloud
        rgbd_img = colordepth_to_rgbd(color_img, depth_img, depth_scale)
        pcd = rgbd_to_pcd(rgbd_img, intrinsics)

        # Scale the pointcloud
        pcd.scale(cb_scale, center=[0,0,0])
        pcds_full.append(pcd)
    return pcds_full

# Align and downsample point clouds
def align_pcds(pcds, c2w_exts, voxel_size=None):
    pcds_align = []
    for i in range(len(pcds)):
        pcd = copy.deepcopy(pcds[i])
        pcd.transform(c2w_exts[i])
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds_align.append(pcd)
    return pcds_align



# Remove outliers from point cloud
def remove_outliers(pcd, nb_neighbors=None, std_ratio=None, 
        nb_points=None, radius=None):
    # Statistical outlier removal
    if nb_neighbors is not None and std_ratio is not None:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
    # Radius outlier removal
    if nb_points is not None and radius is not None:
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=nb_points,
            radius=radius
        )
    return pcd



########### CAMERA CALIBRATION ###########
# Finetune chessboard corners
def finetune_corners(gray, corners, criteria, size=6):
    corners2 = cv2.cornerSubPix(gray,corners,(size, size),(-1,-1),criteria)
    # Arrange corner order
    x1, y1 = corners2[0, 0]
    x2, y2 = corners2[-1, 0]
    flip = False
    if x1 > x2 or y1 < y2:
        corners2 = corners2[::-1]
        flip = True
    return corners2, flip


########### PLOTTING AND VISUALIZATION ###########
# Makes plt grid of images
def show_imgs(imgs, titles=None, width=2):
    if titles is not None: assert len(imgs) == len(titles)
    f = lambda x: (x + width - 1) // width
    n = len(imgs)
    fig, axs = plt.subplots(f(n), width, figsize=(16, 6*f(n)))
    for i, ax in enumerate(axs.flatten()):
        if i >= n: break
        ax.imshow(imgs[i])
        if titles is not None: 
            ax.set_title(titles[i])
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# Draw origin axis from chessboard corners on image
def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    corner = (int(corner[0]), int(corner[1]))
    imgpts = np.int32(imgpts).reshape(-1,2)

    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# Generate Cameras from c2w_exts
def gen_cameras(c2w_exts, camera_size=0.1):
    cameras = []
    for c2w_ext in c2w_exts:
        # Create a camera frame
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=camera_size, origin=[0,0,0])
        camera.transform(c2w_ext)
        cameras.append(camera)
    return cameras






if __name__ == "__main__":
    depth_ints, color_ints = get_o3d_intrinsics()
    print(depth_ints.intrinsic_matrix)
    print(color_ints.intrinsic_matrix)
    