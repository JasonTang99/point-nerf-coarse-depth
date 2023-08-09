# Based on https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/read_bag_example.py

import pyrealsense2 as rs
import numpy as np
import cv2
import os.path
import time
import math
import argparse
import json
import matplotlib.pyplot as plt

class AppState:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]

# Argparse for file path
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path to the bag file")
args = parser.parse_args()

npy_fp = args.file.replace('.bag', '.npy')

# Load npy file
depth_map = np.load(npy_fp, allow_pickle=True)
print(depth_map.shape)

# Load pc file
pc = np.load("pc_" + npy_fp, allow_pickle=True)
print(pc.shape)

# exit(0)

# Take first 10 frames and average
# depth_map = depth_map[:10]
# depth_map, confidence = np.mean(depth_map, axis=0), np.var(depth_map, axis=0)

# depth_frame = 


# TODO  -  Remove this line
depth_map = depth_map[69] / 1000
# subsample
# depth_map = depth_map[::2, ::2]
confidence = np.ones(depth_map.shape)

# Sum > 60000 entries
print(np.sum(depth_map > 65000))

h, w = depth_map.shape

# imshow
# plt.imshow(depth_map)
# plt.show()
# exit(0)


print(depth_map.shape, confidence.shape)
print(np.max(depth_map), np.min(depth_map), np.mean(depth_map))
print(np.max(confidence), np.min(confidence), np.mean(confidence))
# exit(0)
# Load intrinsic parameters in intrinsics.json
with open('intrinsics.json') as f:
    intrinsics_values = json.load(f)
intrinsics = rs.intrinsics()
intrinsics.width = intrinsics_values['width']
intrinsics.height = intrinsics_values['height']
intrinsics.ppx = intrinsics_values['ppx']
intrinsics.ppy = intrinsics_values['ppy']
intrinsics.fx = intrinsics_values['fx']
intrinsics.fy = intrinsics_values['fy']
intrinsics.coeffs = intrinsics_values['coeffs']
intrinsics.model = rs.distortion.none

depth_intrinsics = intrinsics

print(intrinsics_values)
# exit(0)

# result = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
# x, y, z = result[2], -result[0], -result[1]
# Iterate over depth map and get 3D points
verts = np.zeros((h * w, 3), dtype=np.float32)
for i in range(h):
    for j in range(w):
        result = rs.rs2_deproject_pixel_to_point(intrinsics, [i, j], depth_map[i][j])

        center_pixel =  [intrinsics.ppy/2, intrinsics.ppx/2]
        result_center = rs.rs2_deproject_pixel_to_point(intrinsics, center_pixel, depth_map[i][j])

        verts[i * w + j][0] = result[1] # + result_center[1] # result[1]
        verts[i * w + j][1] = (result[0]- result_center[0]) # result[0]
        verts[i * w + j][2] = result[2]
        
# Save 3D points
np.save("verts_" + npy_fp, verts)
print(verts.shape)

other_verts = pc[0]
print(other_verts.shape)

# compare the 2
# for i in range(verts.shape[0]):
#     if np.linalg.norm(verts[i] - other_verts[i]) > 0.01:
#         print(verts[i], other_verts[i])
#         print(np.linalg.norm(verts[i] - other_verts[i]))
#     else:
#         print("OK")

# exit(0)

# Generate texcoords (H, W, 2)
texcoords = np.zeros((h, w, 2), dtype=np.float32)
texcoords[:, :, 0] = np.linspace(0, 1, h)[:, np.newaxis]
texcoords[:, :, 1] = np.linspace(0, 1, w)[np.newaxis, :]
texcoords = texcoords.reshape(-1, 2)

# Generate color map (H, W, 3)
color_source = np.ones((h, w, 3), dtype=np.uint8) * 255

# load points.npy
other_verts = np.load("points.npy")

print(np.min(verts[:, 0]), np.max(verts[:, 0]), np.mean(verts[:, 0]))
print(np.min(verts[:, 1]), np.max(verts[:, 1]), np.mean(verts[:, 1]))
print(np.min(verts[:, 2]), np.max(verts[:, 2]), np.mean(verts[:, 2]))
print("---")
print(np.min(other_verts[:, 0]), np.max(other_verts[:, 0]), np.mean(other_verts[:, 0]))
print(np.min(other_verts[:, 1]), np.max(other_verts[:, 1]), np.mean(other_verts[:, 1]))
print(np.min(other_verts[:, 2]), np.max(other_verts[:, 2]), np.mean(other_verts[:, 2]))


# compare allclose
print(np.allclose(verts, other_verts))
for i in range(100):
    if not np.allclose(verts[i], other_verts[i]):
        print(i, verts[i], other_verts[i])

# exit(0)

# Setup State
state = AppState()
state.color = False
state.decimate = 0


# Create opencv window to render image in
cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)

out = np.empty((h, w, 3), dtype=np.uint8)

# Render
while True:
    now = time.time()
    out.fill(0)

    grid(out, (0, 0.5, 1), size=1, n=10)
    frustum(out, depth_intrinsics)
    axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

    if not state.scale or out.shape[:2] == (h, w):
        pointcloud(out, verts, texcoords, color_source)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, color_source)
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    if any(state.mouse_btns):
        axes(out, view(state.pivot), state.rotation, thickness=4)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

    if key == ord("r"):
        state.reset()

    if key == ord("p"):
        state.paused ^= True

    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

    if key == ord("z"):
        state.scale ^= True

    if key == ord("c"):
        state.color ^= True

    if key == ord("s"):
        cv2.imwrite('./out.png', out)

    if key == ord("e"):
        points.export_to_ply('./out.ply', mapped_frame)

    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break
