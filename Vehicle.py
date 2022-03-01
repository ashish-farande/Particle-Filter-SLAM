import os

from ParticleFilter import *
from Sensors.Lidar import Lidar
from Sensors.DifferentialDrive import DifferentialDrive
from Sensors.Stereo import Stereo
from Sensors.sensor_utils import *

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# The Vehicle will know details about the sensors and its data
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
SENSOR_DATA_PATH = os.path.join(DATA_PATH, 'sensor_data')
IMAGES_DATA_PATH = os.path.join(DATA_PATH, 'stereo_images')
PARAM_PATH = os.path.join(DATA_PATH, 'param')

LIDAR_DATA_FILE = os.path.join(SENSOR_DATA_PATH, 'lidar.csv')
FOG_DATA_FILE = os.path.join(SENSOR_DATA_PATH, 'fog.csv')
ENCODER_DATA_FILE = os.path.join(SENSOR_DATA_PATH, 'encoder.csv')
LEFT_CAMERA_DATA_PATH = os.path.join(IMAGES_DATA_PATH, 'stereo_left')
RIGHT_CAMERA_DATA_PATH = os.path.join(IMAGES_DATA_PATH, 'stereo_right')


LIDAR_TO_VEHICLE_PARAMETERS_PATH = os.path.join(PARAM_PATH, 'Vehicle2Lidar.txt')
FOG_TO_VEHICLE_PARAMETERS_PATH = os.path.join(PARAM_PATH, 'Vehicle2FOG.txt')
STEREO_TO_VEHICLE_PARAMETERS_PATH = os.path.join(PARAM_PATH, 'Vehicle2Stereo.txt')

LEFT_CAMERA_CONFIG_FILE_PATH = os.path.join(PARAM_PATH, "left_camera.yaml")
RIGHT_CAMERA_CONFIG_FILE_PATH = os.path.join(PARAM_PATH, "right_camera.yaml")


class Vehicle:
    def __init__(self, enable_texture_mapping=False):
        self.is_texture_mapping = enable_texture_mapping

        # Create Sensors
        self.lidar = Lidar(param_file=LIDAR_TO_VEHICLE_PARAMETERS_PATH)
        self.motion_sensor = DifferentialDrive(param_file=FOG_TO_VEHICLE_PARAMETERS_PATH)
        self.stereo = Stereo(LEFT_CAMERA_CONFIG_FILE_PATH, RIGHT_CAMERA_CONFIG_FILE_PATH, STEREO_TO_VEHICLE_PARAMETERS_PATH)

        # Create Particle Filter
        self.pf = ParticleFilter()

    def start(self):
        # Start loading the data for the sensors
        self.lidar.load_data(LIDAR_DATA_FILE)
        self.motion_sensor.load_data(gyro_path=FOG_DATA_FILE, encoder_data=ENCODER_DATA_FILE)
        self.stereo.load_data(LEFT_CAMERA_DATA_PATH, RIGHT_CAMERA_DATA_PATH)

        # Initialise Map
        self.pf.initialise_map(self.observe())
        # pf.show_map()

        pass

    def drive(self):
        move_ts = self.motion_sensor.get_next_timestamp()
        lidar_ts = self.lidar.get_next_timestamp()
        stereo_ts = self.stereo.get_next_timestamp()

        i = 0
        while move_ts and lidar_ts:

            self.pf.predict(self.move())
            self.pf.update(self.observe())
            self.pf.resample()

            if self.is_texture_mapping:
                while stereo_ts and lidar_ts  and lidar_ts > stereo_ts:
                    coord, pixel = self.stereo.read_sample()
                    stereo_ts = self.stereo.get_next_timestamp()
                    self.pf.texture_map(coord, pixel)
            self.pf.resample()

            if i % 100 == 0:
                self.pf.show_map( self.is_texture_mapping)

            move_ts = self.motion_sensor.get_next_timestamp()
            lidar_ts = self.lidar.get_next_timestamp()
            i += 1
        pass

    def move(self):
        # Motion
        d_robot_pose = np.array([0.0, 0.0])
        move_ts = self.motion_sensor.get_next_timestamp()
        lidar_ts = self.lidar.get_next_timestamp()

        while move_ts < lidar_ts:
            d_robot_pose += self.motion_sensor.read_sample()
            move_ts = self.motion_sensor.get_next_timestamp()
        return d_robot_pose

    def observe(self):
        lidar_data = self.lidar.read_sample()
        angles, ranges = get_valid(LIDAR_ANGLES, lidar_data)
        coord = convert_angle_coord(angles, ranges)
        s_b = self.lidar.convert_to_body_frame(coord)
        return s_b

    def show_map(self):
        self.pf.map.show_map()

