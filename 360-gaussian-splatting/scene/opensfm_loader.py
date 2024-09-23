#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import collections
import struct
import math
import os
from pyproj import Proj

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params", "panorama"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids", "diff_ref"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def angle_axis_to_quaternion(angle_axis: np.ndarray):
    angle = np.linalg.norm(angle_axis)

    x = angle_axis[0] / angle
    y = angle_axis[1] / angle
    z = angle_axis[2] / angle

    qw = math.cos(angle / 2.0)
    qx = x * math.sqrt(1 - qw * qw)
    qy = y * math.sqrt(1 - qw * qw)
    qz = z * math.sqrt(1 - qw * qw)

    return np.array([qw, qx, qy, qz])

def angle_axis_and_angle_to_quaternion(angle, axis):
    half_angle = angle / 2.0
    sin_half_angle = math.sin(half_angle)
    return np.array([
        math.cos(half_angle),
        axis[0] * sin_half_angle,
        axis[1] * sin_half_angle,
        axis[2] * sin_half_angle
    ])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([w, x, y, z])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_opensfm_points3D(reconstructions):
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    for reconstruction in reconstructions:
        num_points = num_points + len(reconstruction["points"])

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    #reference_lat_0 = reconstructions[0]["reference_lla"]["latitude"]
    #reference_lon_0 = reconstructions[0]["reference_lla"]["longitude"]
    #reference_alt_0 = reconstructions[0]["reference_lla"]["altitude"]
    #e2u_zone=int(divmod(reference_lon_0, 6)[0])+31
    #e2u_conv=Proj(proj='utm', zone=e2u_zone, ellps='WGS84')
    reference_x_0, reference_y_0 = 0 , 0 
    reference_alt_0  = 0
    for reconstruction in reconstructions:
        #reference_lat = reconstruction["reference_lla"]["latitude"]
        #reference_lon = reconstruction["reference_lla"]["longitude"]
        #reference_alt = reconstruction["reference_lla"]["altitude"]
        reference_x, reference_y = 0 , 0
        reference_alt = 0
        for i in (reconstruction["points"]):
            color = (reconstruction["points"][i]["color"])
            coordinates = (reconstruction["points"][i]["coordinates"])
            xyz = np.array([coordinates[0] + reference_x - reference_x_0, coordinates[1] + reference_y - reference_y_0, coordinates[2] - reference_alt + reference_alt_0])
            rgb = np.array([color[0], color[1], color[2]])
            error = np.array(0)
            xyzs[count] = xyz
            rgbs[count] = rgb
            errors[count] = error
            count += 1

    return xyzs, rgbs, errors

def read_opensfm_intrinsics_split(reconstructions):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    cam_id = 4 # 0 ~ 3 are for panorama cameras
    for reconstruction in reconstructions:
        for camera in reconstruction["cameras"]:
            if reconstruction["cameras"][camera]['projection_type'] == 'spherical' or reconstruction["cameras"][camera]['projection_type'] == 'equirectangular':
                model = "SIMPLE_PINHOLE"
                width = reconstruction["cameras"][camera]["width"] / 4
                height = width#econstruction["cameras"][camera]["height"]
                f = width / 2# assume fov = 90
                params = np.array([f, width , height])
                orientation = ["front", "left", "back", "right"]
                for j in range(len(orientation)):
                    camera_id = j
                    cameras[camera_id] = Camera(id=camera_id, model=model,
                                                width=width, height=height,
                                                params=params, panorama=False)
            elif reconstruction["cameras"][camera]['projection_type'] == "perspective":
                model = "SIMPLE_PINHOLE"
                width = reconstruction["cameras"][camera]["width"]
                height = reconstruction["cameras"][camera]["height"]
                f = reconstruction["cameras"][camera]["focal"]
                k1 = reconstruction["cameras"][camera]["k1"]
                k2 = reconstruction["cameras"][camera]["k2"]
                params = np.array([f, width / 2, width / 2, k1, k2])
                camera_id = cam_id
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params, panorama=False)
                cam_id += 1
    return cameras

def read_opensfm_intrinsics(reconstructions):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    cam_id = 1
    for reconstruction in reconstructions:
        for i, camera in enumerate(reconstruction["cameras"]):
            if reconstruction["cameras"][camera]['projection_type'] == 'spherical' or reconstruction["cameras"][camera]['projection_type'] == 'equirectangular':
                camera_id = 0 # assume only one camera
                model = "SPHERICAL"
                width = reconstruction["cameras"][camera]["width"]
                height = width / 4
                f = 0
                params = np.array([f, width , height])
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params, panorama=True)
            elif reconstruction["cameras"][camera]['projection_type'] == "perspective":
                model = "SIMPLE_PINHOLE"
                width = reconstruction["cameras"][camera]["width"]
                height = reconstruction["cameras"][camera]["height"]
                f = reconstruction["cameras"][camera]["focal"]
                k1 = reconstruction["cameras"][camera]["k1"]
                k2 = reconstruction["cameras"][camera]["k2"]
                params = np.array([f, width / 2, width / 2, k1, k2])
                camera_id = cam_id
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params, panorama=False)
                cam_id += 1
    return cameras


def read_opensfm_extrinsics_split(reconstructions):
    images = {}
    i = 0
    for reconstruction in reconstructions:
        for shot in reconstruction["shots"]:
            if reconstruction["cameras"][reconstruction["shots"][shot]["camera"]]['projection_type'] == 'spherical' or reconstruction["cameras"][reconstruction["shots"][shot]["camera"]]['projection_type'] == 'equirectangular':
                translation = reconstruction["shots"][shot]["translation"]
                rotation = reconstruction["shots"][shot]["rotation"]
                qvec = angle_axis_to_quaternion(rotation)
                tvec = np.array([translation[0], translation[1], translation[2]])
                orientation = ["front", "left", "back", "right"]
                for j in range(len(orientation)): 
                    image_id = i
                    camera_id = j
                    shot_with_orientation = os.path.splitext(shot)[0]
                    image_name = shot_with_orientation + orientation[j] + ".jpg"
                    xys = np.array([0, 0]) # dummy
                    point3D_ids = np.array([0, 0]) # dummy
                    images[image_id] = Image(
                        id=image_id, qvec=qvec, tvec=tvec,
                        camera_id=camera_id, name=image_name,
                        xys=xys, point3D_ids=point3D_ids)
                    i += 1
            elif reconstruction["cameras"][reconstruction["shots"][shot]["camera"]]['projection_type'] == 'perspective':
                translation = reconstruction["shots"][shot]["translation"]
                rotation = reconstruction["shots"][shot]["rotation"]
                qvec = angle_axis_to_quaternion(rotation)
                tvec = np.array([translation[0], translation[1], translation[2]])
                image_id = i
                camera_id =4#reconstruction["cameras"]["camera"]['projection_type']
                shot_with_orientation = os.path.splitext(shot)[0]
                image_name = shot
                xys = np.array([0, 0]) # dummy
                point3D_ids = np.array([0, 0]) # dummy
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
                i += 1
    return images

def read_opensfm(reconstructions):
    images = {}
    i = 0
    #reference_lat_0 = reconstructions[0]["reference_lla"]["latitude"]
    #reference_lon_0 = reconstructions[0]["reference_lla"]["longitude"]
    #reference_alt_0 = reconstructions[0]["reference_lla"]["altitude"]
    #e2u_zone=int(divmod(reference_lon_0, 6)[0])+31
    #e2u_conv=Proj(proj='utm', zone=e2u_zone, ellps='WGS84')
    reference_x_0, reference_y_0 = 0 , 0
    reference_alt_0 = 0
    cameras = {}
    camera_names = {}
    cam_id = 1
    for reconstruction in reconstructions:
        for i, camera in enumerate(reconstruction["cameras"]):
            camera_name = camera
            camera_info = reconstruction["cameras"][camera]
            if camera_info['projection_type'] in ['spherical', 'equirectangular']:
                camera_id = 0
                model = "SPHERICAL"
                width = reconstruction["cameras"][camera]["width"]
                height = reconstruction["cameras"][camera]["height"]
                f = width / 4 / 2# assume fov = 90
                params = np.array([f, width , height])
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params, panorama=True)
                camera_names[camera_name] = camera_id
            elif reconstruction["cameras"][camera]['projection_type'] == "perspective":
                model = "SIMPLE_PINHOLE"
                width = reconstruction["cameras"][camera]["width"]
                height = reconstruction["cameras"][camera]["height"]
                f = reconstruction["cameras"][camera]["focal"] * width
                k1 = reconstruction["cameras"][camera]["k1"]
                k2 = reconstruction["cameras"][camera]["k2"]
                params = np.array([f, width / 2, width / 2, k1, k2])
                camera_id = cam_id
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params, panorama=False)
                camera_names[camera_name] = camera_id
                cam_id += 1
    for reconstruction in reconstructions:
        #reference_lat = reconstruction["reference_lla"]["latitude"]
        #reference_lon = reconstruction["reference_lla"]["longitude"]
        #reference_alt = reconstruction["reference_lla"]["altitude"]
        reference_x, reference_y = 0 , 0 
        reference_alt = 0
        for shot in reconstruction["shots"]:
            translation = reconstruction["shots"][shot]["translation"]
            rotation = reconstruction["shots"][shot]["rotation"]
            qvec = angle_axis_to_quaternion(rotation)
            diff_ref_x = reference_x - reference_x_0
            diff_ref_y = reference_y - reference_y_0
            diff_ref_alt = reference_alt - reference_alt_0
            tvec = np.array([translation[0], translation[1], translation[2]])
            diff_ref = np.array([diff_ref_x, diff_ref_y, diff_ref_alt])
            camera_name = reconstruction["shots"][shot]["camera"] 
            camera_id = camera_names.get(camera_name, 0)  # カメラ名からIDを取得
            image_id = i
            image_name = shot
            xys = np.array([0, 0]) # dummy 
            point3D_ids = np.array([0, 0]) # dummy
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids, diff_ref=diff_ref)
            i += 1
    return cameras, images
