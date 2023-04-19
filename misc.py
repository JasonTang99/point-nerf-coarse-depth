import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json

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


if __name__ == "__main__":
    depth_ints_path = "objects/depth_intrinsics.json"
    color_ints_path = "objects/color_intrinsics.json"
    get_intrinsics(depth_ints_path, color_ints_path)
    