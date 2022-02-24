import os

import matplotlib.pyplot as plt;
import numpy as np
import pandas as pd

plt.ion()

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
SENSOR_DATA_PATH = os.path.join(DATA_PATH, 'sensor_data')
PARAM_PATH = os.path.join(DATA_PATH, 'param')

VEHICLE_TO_LIDAR = 'Vehicle2Lidar.txt'
VEHICLE_TO_FOG = 'Vehicle2FOG.txt'
VEHICLE_TO_STEREO = 'Vehicle2Stereo.txt'

LEFT_WHEEL_DIAMETER = 0.623479
RIGHT_WHEEL_DIAMETER = 0.622806


def convert_angle_coord(angles, distances):
    coord = np.zeros((angles.shape[0], 2))
    coord[:, 0] = distances * np.cos(angles)
    coord[:, 1] = distances * np.sin(angles)
    return coord


class Sensor:
    def __init__(self):
        self.transition_matrix = None
        self.rotation_matrix = None
        self.translation = None
        self.data = None
        self.timestamp = None
        self.next_index = 1

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
        if self.next_index < self.data.shape[0]:
            data = self.data[self.next_index]
            self.next_index += 1
        else:
            print("No more Samples!")
        return data

    def get_next_timestamp(self):
        if self.next_index < self.data.shape[0]:
            return self.timestamp[self.next_index]
        else:
            print("No more Samples!")
        return None


class Lidar(Sensor):
    def __init__(self):
        super().__init__()
        Sensor.get_transition_matrix(self, os.path.join(PARAM_PATH, VEHICLE_TO_LIDAR))

    def disp(self, ranges):
        angles = np.linspace(-5, 185, 286) / 180 * np.pi
        plt.figure()
        ax = plt.subplot(111, projection='polar')
        ax.disp(angles, ranges)
        ax.set_rmax(80)
        ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
        ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
        ax.grid(True)
        ax.set_title("Lidar scan data", va='bottom')
        plt.show()


class Move_Robot(Sensor):
    def __init__(self):
        super().__init__()
        self.encoder = Sensor()
        self.gyro = Sensor()
        self.gyro.get_transition_matrix(os.path.join(PARAM_PATH, VEHICLE_TO_FOG))
        self.transition_matrix = self.gyro.transition_matrix
        self.rotation_matrix = self.gyro.rotation_matrix
        self.translation = self.gyro.translation
        self.gyro_index = 0

    def load_data(self, filename=None, gyro_path=None, encoder_data=None):
        if gyro_path and encoder_data:
            self.gyro.load_data(gyro_path)
            self.encoder.load_data(encoder_data)
        else:
            print("No file name provided")

    def read_sample(self):
        theta = 0
        if self.next_index < self.encoder.data.shape[0]:
            while self.gyro_index < self.gyro.data.shape[0] and self.gyro.timestamp[self.gyro_index] < self.encoder.timestamp[self.next_index]:
                theta += self.gyro.data[self.gyro_index, 2]
                self.gyro_index += 1

            ticks = self.encoder.data[self.next_index] - self.encoder.data[self.next_index - 1]
            left_dist = np.pi * LEFT_WHEEL_DIAMETER * ticks[0] / (4096.0 * 1)
            right_dist = np.pi * RIGHT_WHEEL_DIAMETER * ticks[1] / (4096.0 * 1)
            dist = (left_dist + right_dist) / 2

            self.next_index += 1
            return dist, theta
        else:
            print("No more Samples!")
        return None

    def get_next_timestamp(self):
        if self.next_index < self.encoder.data.shape[0] and self.gyro_index < self.gyro.data.shape[0]:
            return self.encoder.timestamp[self.next_index]
        else:
            print("No more Samples!")
        return None
