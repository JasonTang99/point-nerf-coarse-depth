import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import open3d as o3d

# Gets the intrinsics of depth and color streams
def get_intrinsics(depth_ints_path, color_ints_path):
    pipeline = rs.pipeline()
    config = rs.config()

    bag_fp = "objects/color_sample.bag"
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


# Load o3d intrinsics
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

# Get o3d intrinsics
def get_o3d_intrinsics():
    depth_path = "data/intrinsics/depth_intrinsics.json"
    color_path = "data/intrinsics/color_intrinsics.json"
    depth_intrinsics = _read_intrinsics(depth_path)
    color_intrinsics = _read_intrinsics(color_path)
    return depth_intrinsics, color_intrinsics


# Combine color and depth images into a single RGBD image
def colordepth_to_rgbd(color_img, depth_img):
    assert len(color_img.shape) == 3
    assert len(depth_img.shape) == 2 or \
        (len(depth_img.shape) == 3 and depth_img.shape[2] == 1)
    
    color_raw = o3d.geometry.Image(color_img)
    depth_raw = o3d.geometry.Image(depth_img)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False)
    return rgbd_image

# RGBD to point cloud
def rgbd_to_pcd(rgbd_image, intrinsics, extrinsics=np.eye(4), voxel_size=None):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        intrinsics, 
        extrinsics
    )
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


if __name__ == "__main__":
    depth_ints, color_ints = get_o3d_intrinsics()
    print(depth_ints.intrinsic_matrix)
    print(color_ints.intrinsic_matrix)
    