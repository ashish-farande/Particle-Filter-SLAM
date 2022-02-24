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
    def __init__(self, position=[0, 0], angle=0, weight=0.0):
        self.position = position
        self.angle = angle
        self.weight = weight


def get_noise(sigma):
    return np.random.normal(0, sigma, 1)


class ParticleFilter:
    def __init__(self, n_particles=10):
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
        self.drive.load_data(gyro_path=os.path.join(SENSOR_DATA_PATH, FOG_DATA_FILE), encoder_data=os.path.join(SENSOR_DATA_PATH, ENCODER_DATA_FILE))

        self.one_particle = Particle()

    def predict(self):
        d_robot_pose = np.array([0.0, 0.0])
        move_ts = self.drive.get_next_timestamp()
        lidar_ts = self.lidar.get_next_timestamp()

        while move_ts < lidar_ts:
            d_robot_pose += self.drive.read_sample()
            move_ts = self.drive.get_next_timestamp()

        self.one_particle.position[0] += d_robot_pose[0] * np.cos(self.one_particle.angle)
        self.one_particle.position[1] += d_robot_pose[0] * np.sin(self.one_particle.angle)
        self.one_particle.angle += d_robot_pose[1]

        # TODO: Play with noise
        # for particle in self.particles:
        #     particle.position[0] += d_robot_pose[0] * np.cos(particle.angle) + get_noise(0.1)
        #     particle.position[1] += d_robot_pose[0] * np.sin(particle.angle) + get_noise(0.1)
        #     particle.angle += d_robot_pose[1] + get_noise(0.05)

    def update(self):
        # TODO: Map correlation to get the new weights particle
        best_particle = self.one_particle

        # Scan and update using the best particle
        lidar_coord = self.scan()
        lidar_coord_world = convert_to_world_frame(best_particle.angle, best_particle.position, lidar_coord[:, :2])

        self.update_map(lidar_coord_world, best_particle)

    def update_map(self, lidar_coord, best_particle):
        self.map.update_free(best_particle.position, lidar_coord)
        self.map.update_map()
        return

    def scan(self):
        lidar_data = self.lidar.read_sample()
        angles, ranges = get_valid(LIDAR_ANGLES, lidar_data)
        coord = convert_angle_coord(angles, ranges)
        s_b = self.lidar.convert_to_body_frame(coord)
        return s_b

    def initialise_map(self):
        best_particle = self.one_particle

        lidar_coord = self.scan()
        lidar_coord_world = convert_to_world_frame(best_particle.angle, best_particle.position, lidar_coord[:, :2])
        self.update_map(lidar_coord_world, best_particle)
        return

    def show_map(self):
        self.map.display()

    def resample(self):
        # TODO: Resampling
        pass
