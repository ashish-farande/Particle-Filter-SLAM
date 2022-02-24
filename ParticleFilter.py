import copy

from Map import *
from Sensor import *
from sensor_utils import *

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(PROJECT_PATH, 'data')
SENSOR_DATA_PATH = os.path.join(DATA_PATH, 'sensor_data')

LIDAR_DATA_FILE = 'lidar.csv'
FOG_DATA_FILE = 'fog.csv'
ENCODER_DATA_FILE = 'encoder.csv'
LIDAR_ANGLES = np.linspace(-5, 185, 286) / 180 * np.pi


class Particle:
    def __init__(self, position=(0, 0), angle=0, weight=0.0):
        self.position = position
        self.angle = angle
        self.weight = weight


class ParticleFilter:
    def __init__(self, n_particles=100):
        self.particles = []

        self.map = Map()

        # NOTE: Should not be part of Particle filter but for now we will create it inside the particle Filter
        self.lidar = Lidar()
        self.drive = Move_Robot()

        for i in range(n_particles):
            self.particles.append(Particle(weight=1 / n_particles))

        # Let any particle be the best particle
        self.best_particle = self.particles[0]

        # Set Up for the sensors
        self.lidar.load_data(os.path.join(SENSOR_DATA_PATH, LIDAR_DATA_FILE))
        self.move.load_data(gyro_path=os.path.join(SENSOR_DATA_PATH, FOG_DATA_FILE), encoder_data=os.path.join(SENSOR_DATA_PATH, ENCODER_DATA_FILE))

    def predict(self):
        pass

    def update(self):
        pass

    def update_map(self, lidar_coord):
        lidar_coord_world = convert_to_world_frame(self.best_particle.angle, self.best_particle.position, lidar_coord[:, :2])
        self.map.update_free(copy.deepcopy(self.best_particle.position), lidar_coord_world)
        self.map.update_map()
        return

    def scan(self):
        lidar_data = self.lidar.read_sample()
        angles, ranges = get_valid(LIDAR_ANGLES, lidar_data)
        coord = convert_angle_coord(angles, ranges)
        s_b = self.lidar.convert_to_body_frame(coord)
        return s_b

    def add_noise(self):
        pass

    def initialise_map(self):
        self.update_map(self.scan())
        return
