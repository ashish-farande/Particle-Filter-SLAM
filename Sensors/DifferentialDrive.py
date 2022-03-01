import numpy as np
from Sensors.Sensor import Sensor

LEFT_WHEEL_DIAMETER = 0.623479
RIGHT_WHEEL_DIAMETER = 0.622806


class DifferentialDrive(Sensor):
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
