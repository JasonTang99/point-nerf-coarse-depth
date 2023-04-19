import pyrealsense2 as rs
import numpy as np
import cv2
import os

def get_aligned_images(
        bag_fp = "objects/color_sample.bag",
        frame_idx = 0,
        color_path = "objects/color.npy",
        depth_path = "objects/depth.npy",
        plot = False
    ):
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

    # Create an align object
    align = rs.align(rs.stream.color)

    # Write to file paths
    try:
        depth_image, color_image = None, None
        for i in range(frame_idx+1):
            # Align the depth frame to color frame
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if plot:
                # Set pixels further than clipping_distance to grey
                grey_color = 153
                depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

                # Render images
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                images = np.hstack((bg_removed, depth_colormap))

                cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
                cv2.imshow('Align Example', images)
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
        # Save images
        if not depth_image is None and not color_image is None:
            print("Saving images to {} and {}".format(color_path, depth_path))
            np.save(color_path, color_image)
            np.save(depth_path, depth_image)
    finally:
        pipeline.stop()

if __name__ == "__main__":
    get_aligned_images(
        bag_fp = "objects/color_sample.bag",
        frame_idx = 0,
        color_path = "objects/color.npy",
        depth_path = "objects/depth.npy",
        plot = False
    )