from Map import *
from Sensor import *
from sensor_utils import *

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

LIDAR_ANGLES = np.linspace(-5, 185, 286) / 180 * np.pi


def add_noise(n, sigmas):
    n = np.random.normal(np.zeros(len(sigmas)), sigmas, size=(n, len(sigmas)))
    return n


class Particle:
    def __init__(self, position=[0, 0], angle=0, weight=0.0):
        self.position = position
        self.angle = angle
        self.weight = weight


class ParticleFilter:
    def __init__(self, n_particles=20, lidar_sensor=None, motion_sensor=None, enable_one_particle=False):
        assert lidar_sensor is not None
        assert motion_sensor is not None

        self.n_particles = n_particles
        self.particles = []
        self.enable_one_particle = enable_one_particle

        self.map = Map()

        # NOTE: Should not be part of Particle filter but for now we will create it inside the particle Filter
        self.lidar = lidar_sensor
        self.drive = motion_sensor

        for i in range(n_particles):
            self.particles.append(Particle(weight=1 / n_particles))

        # Let any particle be the best particle
        self.best_particle = Particle()

        self.one_particle = Particle()

        self.particle_poses = np.zeros((n_particles, 3))
        self.particle_weights = np.ones(n_particles) / n_particles

    def predict(self):
        d_robot_pose = np.array([0.0, 0.0])
        move_ts = self.drive.get_next_timestamp()
        lidar_ts = self.lidar.get_next_timestamp()

        while move_ts < lidar_ts:
            d_robot_pose += self.drive.read_sample()
            move_ts = self.drive.get_next_timestamp()

        if self.enable_one_particle:
            self.one_particle.position[0] += d_robot_pose[0] * np.cos(self.one_particle.angle)
            self.one_particle.position[1] += d_robot_pose[0] * np.sin(self.one_particle.angle)
            self.one_particle.angle += d_robot_pose[1]

        else:
            # TODO: Play with noise
            for i in range(self.particle_poses.shape[0]):
                self.particle_poses[i, 0] += d_robot_pose[0] * np.cos(self.particle_poses[i, 2])
                self.particle_poses[i, 1] += d_robot_pose[0] * np.sin(self.particle_poses[i, 2])
                self.particle_poses[i, 2] += d_robot_pose[1]

            self.particle_poses[:-1] += add_noise(self.particle_poses.shape[0] - 1, [0.5, 0.5, 0.01])

    def update(self):
        lidar_coord = self.scan()

        if self.enable_one_particle:
            self.best_particle = self.one_particle
        else:
            # TODO: MAP Correlation
            max_weight = 0
            for i in range(self.particle_poses.shape[0]):
                lidar_coord_particle = convert_to_world_frame(self.particle_poses[i, 2], self.particle_poses[i, :2], lidar_coord[:, :2])
                self.particle_weights[i] = self.map.map_correlation(self.particle_poses[i, :2], lidar_coord_particle)
                if self.particle_weights[i] > max_weight:
                    max_weight = self.particle_weights[i]

            self.particle_weights /= np.sum(self.particle_weights)
            best_particle_pose = self.particle_poses[np.argmax(self.particle_weights)]
            self.best_particle = Particle(position=best_particle_pose[:2], angle=best_particle_pose[2], weight=max_weight)

        # Scan and update using the best particle
        lidar_coord_world = convert_to_world_frame(self.best_particle.angle, self.best_particle.position, lidar_coord[:, :2])

        self.update_map(lidar_coord_world, self.best_particle)

    def update_map(self, lidar_coord, best_particle):
        self.map.update_free(np.array(best_particle.position), lidar_coord)
        # self.map.update_map()
        return

    def scan(self):
        lidar_data = self.lidar.read_sample()
        angles, ranges = get_valid(LIDAR_ANGLES, lidar_data)
        coord = convert_angle_coord(angles, ranges)
        s_b = self.lidar.convert_to_body_frame(coord)
        return s_b

    def initialise_map(self):
        self.best_particle = Particle()

        lidar_coord = self.scan()
        lidar_coord_world = convert_to_world_frame(self.best_particle.angle, self.best_particle.position, lidar_coord[:, :2])
        self.update_map(lidar_coord_world, self.best_particle)
        return

    def show_map(self):
        self.map.display()

    def resample(self):
        # TODO: Resampling
        # n_eff = 1/(np.sum(self.particle_weights**2))
        # if n_eff < :
        # sort the weight vector
        sorted_index = np.argsort(self.particle_weights)
        sorted_particle_poses = self.particle_poses[sorted_index]

        particle_cum = np.cumsum(self.particle_weights)
        new_poses = []
        for i in range(self.n_particles - 1):
            n_ = np.random.uniform(0, 1)
            j = 0
            while n_ > particle_cum[j]:
                j += 1
            new_poses.append(sorted_particle_poses[j, :])
        self.particle_poses[:-1] = np.array(new_poses)
        self.particle_weights = np.ones(self.n_particles) / self.n_particles

    def texture_map(self, coord, pixel_values):
        stereo_coord_world = convert_to_world_frame(self.best_particle.angle, self.best_particle.position, coord[:, :2])
        self.map.build_texture(stereo_coord_world, pixel_values)
