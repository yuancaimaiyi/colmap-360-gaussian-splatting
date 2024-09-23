import os
import cv2 
import json
import glob
import numpy as np
import argparse

def undistort(data_dir):
    with open(data_dir + '/reconstruction_distorted.json', 'r') as file:
        json_data = json.load(file)
    for i in range(len(json_data)):
        new_cameras = {}
        cam_data = json_data[i]['cameras']
        for camera_name, params in cam_data.items():
            camera_info = {
                'name': camera_name,
                'projection_type': params.get('projection_type', ''),
                'width': params.get('width', 0),
                'height': params.get('height', 0),
                'focal': params.get('focal', 0),
                'k1': params.get('k1', 0),
                'k2': params.get('k2', 0)
            }
            f = camera_info["focal"] * camera_info["width"]
            cx = camera_info["width"] / 2
            cy = camera_info["height"] / 2
            k1 = camera_info["k1"]
            k2 = camera_info["k2"]

            camera_matrix = np.array([[f, 0, cx],
                                    [0, f, cy],
                                    [0, 0, 1]])
            dist_coeffs = np.array([k1, k2, 0, 0, 0])  # k3, p1, p2は0と仮定

            distorted_image_path_list = sorted(glob.glob(data_dir + '/images_distorted/*.*'))
            output_dir = data_dir + '/images_split/'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            for distorted_image_path in distorted_image_path_list:
                try:
                    image = cv2.imread(distorted_image_path)

                    if image is None:
                        raise FileNotFoundError(f"image not found: {distorted_image_path}")

                    # undistort
                    h, w = image.shape[:2]
                    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
                    undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
                    x, y, w, h = roi

                    image_path = output_dir + os.path.basename(distorted_image_path)
                    cv2.imwrite(image_path, undistorted_img[y:y+h, x:x+w])
                except FileNotFoundError as e:
                    print(e)
                except Exception as e:
                    print(f"error:  {e}")
            new_cam_info = camera_info
            new_cam_info['focal'] = new_camera_matrix[0][0]
            new_cam_info['k1'] = 0
            new_cam_info['k2'] = 0
            new_cam_info['width'] = w
            new_cam_info['height'] = h
            new_cameras[camera_name] = new_cam_info
            temp_json = {}
            for camera_name, camera_info in json_data[0]['cameras'].items():
                temp_json[camera_name] = new_cameras[camera_name]
            json_data[0]['cameras'] = temp_json
            with open(data_dir + '/reconstruction.json', 'w') as outfile:
                json.dump(json_data, outfile, indent=4)

                

def main():
    parser = argparse.ArgumentParser(description="Convert equirectangular panorama to cube map.")
    parser.add_argument("data_dir", type=str, help="Input data directory.")

    args = parser.parse_args()

    undistort(args.data_dir)

if __name__ == "__main__":
    main()