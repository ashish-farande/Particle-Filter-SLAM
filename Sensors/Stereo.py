import os
import cv2
import numpy as np

from Sensors.Sensor import Sensor


Z_MIN = -10
Z_MAX = 50


class Camera:
    def __init__(self, config_file):
        config = cv2.FileStorage(config_file, cv2.FILE_STORAGE_READ)
        self.project_matrix = config.getNode("projection_matrix").mat()
        self.inverse_projection_matrix = np.linalg.inv(self.project_matrix[:3, :3])

    def convert_to_optical_frame(self, coord_list):
        new_coord = coord_list @ self.inverse_projection_matrix.T
        return new_coord


class Stereo(Sensor):
    def __init__(self, left_camera_config_file, right_camera_config_file, param_file):
        super().__init__(param_file)
        self.left_camera_data_folder = None
        self.right_camera_data_folder = None
        self.left_camera = Camera(left_camera_config_file)
        self.right_camera = Camera(right_camera_config_file)
        self.depth_constant = (self.left_camera.project_matrix - self.right_camera.project_matrix)[0, 3]

    def load_data(self, folder_path1, folder_path2):
        self.data = None
        self.left_camera_data_folder = folder_path1
        self.right_camera_data_folder = folder_path2
        left_ts = np.sort([f[:-4] for f in os.listdir(folder_path1) if os.path.isfile(os.path.join(folder_path1, f))])
        right_ts = np.sort([f[:-4] for f in os.listdir(folder_path2) if os.path.isfile(os.path.join(folder_path2, f))])
        self.timestamp = left_ts[left_ts == right_ts].astype(int)

    def read_sample(self):
        assert self.left_camera_data_folder is not None
        assert self.right_camera_data_folder is not None
        img = None
        if self.current_index < self.timestamp.shape[0]:
            filename = str(self.timestamp[self.current_index])+".png"
            left_img = cv2.imread(os.path.join(self.left_camera_data_folder, filename), 0)
            right_img = cv2.imread(os.path.join(self.right_camera_data_folder, filename),0 )

            left_img_colour = cv2.cvtColor(left_img, cv2.COLOR_BAYER_BG2BGR)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BAYER_BG2BGR)

            left_img = cv2.cvtColor(left_img_colour, cv2.COLOR_BGR2GRAY)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
            stereo = cv2.StereoBM_create(numDisparities=32, blockSize=9)
            disparity = stereo.compute(left_img, right_img)

            depth = self.depth_constant / disparity
            xs, ys = np.where((depth>0)&(depth != np.inf))

            # TODO: Improvise below code conversion
            coords = np.stack((xs, ys, np.ones(xs.shape[0]))).T
            optical_coords = self.left_camera.convert_to_optical_frame(coords)
            new_optical_coords = depth[xs, ys][:, np.newaxis]*optical_coords
            regular_coord = self.convert_to_body_frame(new_optical_coords)
            pixel_values = left_img_colour[xs, ys]
            z_coord = regular_coord[:,2]
            ind = np.where((z_coord > Z_MIN) & (z_coord < Z_MAX))
            final_coord = regular_coord[ind]
            pixel_values = pixel_values[ind]
            self.current_index += 1
            return final_coord, pixel_values
        else:
            print("No more Samples!")
        return None
