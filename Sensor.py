import os

import matplotlib.pyplot as plt;
import numpy as np
import pandas as pd

plt.ion()

LEFT_WHEEL_DIAMETER = 0.623479
RIGHT_WHEEL_DIAMETER = 0.622806


def convert_angle_coord(angles, distances):
    coord = np.zeros((angles.shape[0], 2))
    coord[:, 0] = distances * np.cos(angles)
    coord[:, 1] = distances * np.sin(angles)
    return coord


class Sensor:
    def __init__(self, param_file=None):
        self.transition_matrix = None
        self.rotation_matrix = None
        self.translation = None
        self.data = None
        self.timestamp = None
        self.current_index = 0
        if param_file:
            self.get_transition_matrix(param_file)

    def load_data(self, filename):
        data_csv = pd.read_csv(filename, header=None)
        self.data = data_csv.values[:, 1:]
        self.timestamp = data_csv.values[:, 0]

    def get_transition_matrix(self, filename):
        with open(filename) as f:
            for line in f:
                if line[0:2] == 'R:':
                    self.rotation_matrix = np.array(line[2:-1].split()).reshape((3, 3)).astype(np.float)
                elif line[0:2] == 'T:':
                    self.translation = np.array(line[2:-1].split()).reshape((3, 1)).astype(np.float)
            transition_matrix = np.hstack((self.rotation_matrix, self.translation))
            self.transition_matrix = np.vstack((transition_matrix, [0, 0, 0, 1])).astype(np.float)

    def convert_to_body_frame(self, data):
        data = np.hstack((data, np.zeros((data.shape[0], 1))))
        new_data = self.rotation_matrix @ data.T + self.translation
        return new_data.T[:, :3]

    def read_sample(self):
        data = None
        if self.current_index < self.data.shape[0]:
            data = self.data[self.current_index]
            self.current_index += 1
        else:
            print("No more Samples!")
        return data

    def get_next_timestamp(self):
        if self.current_index < self.data.shape[0]-1:
            return self.timestamp[self.current_index+1]
        else:
            print("No more Samples!")
        return None


class Lidar(Sensor):
    def __init__(self, param_file):
        super().__init__(param_file)

    # def disp(self, ranges):
    #     angles = np.linspace(-5, 185, 286) / 180 * np.pi
    #     plt.figure()
    #     ax = plt.subplot(111, projection='polar')
    #     ax.disp(angles, ranges)
    #     ax.set_rmax(80)
    #     ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
    #     ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    #     ax.grid(True)
    #     ax.set_title("Lidar scan data", va='bottom')
    #     plt.show()


class Drive(Sensor):
    def __init__(self, param_file):
        super().__init__(param_file)
        self.encoder = Sensor()
        self.gyro = Sensor(param_file)
        self.gyro_index = 0
        self.encoder_current_index = 1

    def load_data(self, filename=None, gyro_path=None, encoder_data=None):
        if gyro_path and encoder_data:
            self.gyro.load_data(gyro_path)
            self.encoder.load_data(encoder_data)
        else:
            print("No file name provided")

    def read_sample(self):
        theta = 0
        if self.encoder_current_index < self.encoder.data.shape[0]:
            while self.gyro_index < self.gyro.data.shape[0] and self.gyro.timestamp[self.gyro_index] < self.encoder.timestamp[self.encoder_current_index]:
                theta += self.gyro.data[self.gyro_index, 2]
                self.gyro_index += 1

            ticks = self.encoder.data[self.encoder_current_index] - self.encoder.data[self.encoder_current_index - 1]
            left_dist = np.pi * LEFT_WHEEL_DIAMETER * ticks[0] / (4096.0 * 1)
            right_dist = np.pi * RIGHT_WHEEL_DIAMETER * ticks[1] / (4096.0 * 1)
            dist = (left_dist + right_dist) / 2

            self.encoder_current_index += 1
            return dist, theta
        else:
            print("No more Samples!")
        return None

    def get_next_timestamp(self):
        if self.encoder_current_index < self.encoder.data.shape[0]-1 and self.gyro_index < self.gyro.data.shape[0]:
            return self.encoder.timestamp[self.encoder_current_index+1]
        else:
            print("No more Samples!")
        return None
