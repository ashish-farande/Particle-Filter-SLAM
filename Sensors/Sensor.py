import numpy as np
import pandas as pd


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
                    self.rotation_matrix = np.array(line[2:-1].split()).reshape((3, 3)).astype(np.float64)
                elif line[0:2] == 'T:':
                    self.translation = np.array(line[2:-1].split()).reshape((3, 1)).astype(np.float64)
            transition_matrix = np.hstack((self.rotation_matrix, self.translation))
            self.transition_matrix = np.vstack((transition_matrix, [0, 0, 0, 1])).astype(np.float64)

    def convert_to_body_frame(self, data):
        if data.shape[1] == 2:
            data = np.hstack((data, np.zeros((data.shape[0], 1))))
        new_data = self.rotation_matrix @ data.T
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
        if self.current_index < self.timestamp.shape[0]-1:
            return self.timestamp[self.current_index+1]
        else:
            print("No more Samples!")
        return None

