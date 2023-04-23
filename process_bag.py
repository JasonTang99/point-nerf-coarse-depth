import pyrealsense2 as rs
import numpy as np
import cv2
import os

def get_images(
        start_idx=0,
        end_idx=None,
        bag_fp="data/aligned/color_sample.bag",
        color_path=None,
        depth_path=None,
        align=True,
        median=True,
        verbose=False
    ):
    """
    Gets color and depth images from a bag file
    """
    if end_idx is None:
        end_idx = start_idx + 1
    
    # Create a pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()

    # Load bag file
    rs.config.enable_device_from_file(
        config, 
        bag_fp,
        repeat_playback=False
    )

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Removing the objects > clipping_distance_in_meters away
    clipping_distance_in_meters = 10
    clipping_distance = clipping_distance_in_meters / depth_scale
    
    # Align frames
    # Set color and depth images
    align_frames = rs.align(rs.stream.color)
    depth_imgs, color_imgs = [], []

    try:
        for i in range(start_idx):
            frames = pipeline.wait_for_frames(timeout_ms=1000)

        for i in range(start_idx, end_idx):
            frames = pipeline.wait_for_frames(timeout_ms=1000)

            if align:
                frames = align_frames.process(frames)

            depth_imgs.append(np.asanyarray(
                frames.get_depth_frame().get_data()
            ).copy())
            color_imgs.append(np.asanyarray(
                frames.get_color_frame().get_data()
            ).copy())

    except Exception as e:
        if verbose:
            print("Error getting frames")
            print(e)
    finally:
        pipeline.stop()
    if verbose:
        print(f"Got {len(depth_imgs)} frames")

    # Stack and save images
    depth_imgs = np.stack(depth_imgs)
    color_imgs = np.stack(color_imgs)

    if color_path is not None and depth_path is not None:
        if verbose:
            print("Saving images to {} and {}".format(color_path, depth_path))
        np.save(color_path, color_imgs)
        np.save(depth_path, depth_imgs)

    if median:
        if verbose:
            print(f"Taking median of {color_imgs.shape[0]} images")
        color_imgs = np.median(color_imgs, axis=0).astype(np.uint8)
        depth_imgs = np.median(depth_imgs, axis=0).astype(np.uint16)
    
    return color_imgs, depth_imgs


if __name__ == "__main__":
    # color_imgs, depth_imgs = get_images(
    #     start_idx = 0,
    #     end_idx = 10,
    #     bag_fp="data/aligned/color_sample.bag",
    #     color_path="data/aligned/color.npy",
    #     depth_path="data/aligned/depth.npy",
    #     align = True,
    #     median = False
    # )

    bag_fp = "data/pose/checkerboard.bag"
    color_path = None
    depth_path = None



    color_imgs, depth_imgs = get_images(
        start_idx=0,
        end_idx=1000,
        bag_fp=bag_fp,
        color_path=color_path,
        depth_path=depth_path,
        align=True,
        median=False
    )

    print("Color images shape: {}".format(color_imgs.shape))
    print("Depth images shape: {}".format(depth_imgs.shape))
