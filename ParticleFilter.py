import os

from Map import *

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

LIDAR_ANGLES = np.linspace(-5, 185, 286) / 180 * np.pi


def convert_to_world_frame(angle, pos, points):
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = points @ rotation.T + pos
    return points


def add_noise(n, sigmas):
    n = np.random.normal(np.zeros(len(sigmas)), sigmas, size=(n, len(sigmas)))
    return n


class Particle:
    def __init__(self, position=[0, 0], angle=0, weight=0.0):
        self.position = position
        self.angle = angle
        self.weight = weight


class ParticleFilter:
    def __init__(self, n_particles=20, enable_one_particle=False):
        self.n_particles = n_particles
        self.enable_one_particle = enable_one_particle

        self.map = Map()

        # Let any particle be the best particle
        self.best_particle = Particle()
        if self.enable_one_particle:
            self.one_particle = Particle()
        else:
            self.particle_poses = np.zeros((n_particles, 3))
            self.particle_weights = np.ones(n_particles) / n_particles

    def predict(self, delta_robot_pose):
        if self.enable_one_particle:
            self.one_particle.position[0] += delta_robot_pose[0] * np.cos(self.one_particle.angle)
            self.one_particle.position[1] += delta_robot_pose[0] * np.sin(self.one_particle.angle)
            self.one_particle.angle += delta_robot_pose[1]
        else:
            # Add noise to the particles
            for i in range(self.particle_poses.shape[0]):
                self.particle_poses[i, 0] += delta_robot_pose[0] * np.cos(self.particle_poses[i, 2])
                self.particle_poses[i, 1] += delta_robot_pose[0] * np.sin(self.particle_poses[i, 2])
                self.particle_poses[i, 2] += delta_robot_pose[1]

            self.particle_poses[:-1] += add_noise(self.particle_poses.shape[0] - 1, [0.5, 0.5, 0.01])

    def update(self, lidar_observation):
        # Find the weights of the particles
        if self.enable_one_particle:
            self.best_particle = self.one_particle
        else:
            # TODO: MAP Correlation
            max_weight = 0
            for i in range(self.particle_poses.shape[0]):
                lidar_coord_particle = convert_to_world_frame(self.particle_poses[i, 2], self.particle_poses[i, :2], lidar_observation[:, :2])
                self.particle_weights[i] = self.map.map_correlation(lidar_coord_particle)
                if self.particle_weights[i] > max_weight:
                    max_weight = self.particle_weights[i]

            # Find the best particle
            self.particle_weights /= np.sum(self.particle_weights)
            best_particle_pose = self.particle_poses[np.argmax(self.particle_weights)]
            self.best_particle = Particle(position=best_particle_pose[:2], angle=best_particle_pose[2], weight=max_weight)

        # Get the world frame coord for the best particle
        lidar_coord_world = convert_to_world_frame(self.best_particle.angle, self.best_particle.position, lidar_observation[:, :2])

        # Update the map using the best particle
        self.update_map(lidar_coord_world, self.best_particle)

    def update_map(self, lidar_coord, best_particle):
        self.map.update_free(np.array(best_particle.position), lidar_coord)
        return

    def initialise_map(self, lidar_observation):
        self.best_particle = Particle()
        lidar_coord_world = convert_to_world_frame(self.best_particle.angle, self.best_particle.position, lidar_observation[:, :2])
        self.update_map(lidar_coord_world, self.best_particle)
        return

    def show_map(self, is_textured):
        self.map.display(is_textured)

    def resample(self):
        # Do the resampling always
        # Stratified Resampling
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
