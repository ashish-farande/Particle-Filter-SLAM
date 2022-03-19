import cv2
import matplotlib.pyplot as plt

from Sensors.sensor_utils import *

MAP_SIZE = 1000

LAMBDA_MIN = -6
LAMBDA_MAX = 6


class Map:
    def __init__(self, x_min=-MAP_SIZE, y_min=-MAP_SIZE - 1000, x_max=MAP_SIZE + 1000, y_max=MAP_SIZE):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.res = 4
        self.x_size = int(np.ceil((x_max - x_min) / self.res + 1))  # cells
        self.y_size = int(np.ceil((y_max - y_min) / self.res + 1))  # cells
        self.odds = np.zeros((self.x_size, self.y_size))
        self.map = np.ones((self.x_size, self.y_size), dtype="uint8")
        self.robot_coord = []
        self.texture_map = None

    def update_free(self, in_start_point, in_end_points):
        """
        Use breshanm to find the free and occupied pixels in the grid map.
        @param in_start_point: Robot position in world frame
        @param in_end_points: Lidar data in world frame
        @return:
        """
        end_points = self.convert_to_map(in_end_points)
        start_point = self.convert_to_map(in_start_point[np.newaxis, :])[0, :]

        # Storing the location to draw the trajectory
        self.robot_coord.append(start_point)

        # Trace the lines of the scans
        free_pixels = get_mapping(start_point, end_points)
        if free_pixels is not None:
            xis, yis = free_pixels[:, 0], free_pixels[:, 1]
            ind_good = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < self.x_size)), (yis < self.y_size))

            # Update the log odds for free
            added_logs = self.odds[xis[ind_good], yis[ind_good]] + np.log(4)
            clipped_vals = np.clip(added_logs, LAMBDA_MIN, LAMBDA_MAX)
            self.odds[xis[ind_good], yis[ind_good]] = clipped_vals

            # Faster way to update the map
            self.update_map(xis[ind_good], yis[ind_good])

        if end_points is not None:
            # Update the log odds for Occupied
            xis, yis = end_points[:, 0], end_points[:, 1]
            ind_good = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < self.x_size)), (yis < self.y_size))
            added_logs = self.odds[xis[ind_good], yis[ind_good]] - np.log(4)
            clipped_vals = np.clip(added_logs, LAMBDA_MIN, LAMBDA_MAX)
            self.odds[xis[ind_good], yis[ind_good]] = clipped_vals
            self.update_map(xis[ind_good], yis[ind_good])

    def update_map(self, xs, ys):
        """
        A internal map used to correlate with the lidar data
        @param xs: x-coordinates to be updated
        @param ys: y-coordinates to be updated
        @return:
        """
        # self.map = np.where(self.odds.T < 0, 1, 0)
        self.map[xs, ys] = np.where(self.odds[xs, ys] < 0, 0, 1)
        pass

    def update_robot_pose(self, robot_position):
        robot_coord = self.convert_to_map(robot_position[np.newaxis, :])[0, :]

        # Storing the location to draw the trajectory
        self.robot_coord.append(robot_coord)

    def map_correlation(self, in_end_points):
        """
        Correlates a set of points with the current map
        @param in_end_points: list of [x,y]
        @return: Correlation Value
        """
        end_points = self.convert_to_map(in_end_points)
        xis, yis = end_points[:, 0].astype(np.int16), end_points[:, 1].astype(np.int16)
        ind_good = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < self.x_size)), (yis < self.y_size))
        val = np.sum(self.map[xis[ind_good], yis[ind_good]])
        return val

    def build_texture(self, coord, pixel_values):
        if self.texture_map is None:
            self.texture_map = np.zeros((self.x_size, self.y_size, 3), dtype="uint8") * 255
        new_coord = self.convert_to_map(coord)
        self.texture_map[new_coord[:, 0], new_coord[:, 1]] = pixel_values

    def convert_to_map(self, points):
        """
        Consert the World frame coordinates to Grid Coordinates
        @param points: list of [x, y]
        @return:
        """
        new_points = np.zeros_like(points)
        new_points[:, 0] = np.ceil((points[:, 0] - self.x_min) / self.res).astype(np.int16) - 1
        new_points[:, 1] = np.ceil((points[:, 1] - self.y_min) / self.res).astype(np.int16) - 1
        return new_points.astype(int)

    def display_map(self, wait_key=10):
        img = ((1 / (1 + np.exp(-self.odds))) * 255).astype(np.uint8)
        name = "Occupancy Map"
        cv2.imshow(name, img)
        cv2.waitKey(wait_key)
        cv2.destroyAllWindows()

    def display_texture_map(self):
        """
        Display the texture map
        @return:
        """
        if self.texture_map is not None:
            img = self.texture_map
            name = "Texture Map"
            cv2.imshow(name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("ERROR: No texture map data available")

    def show_robot_path(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        self.robot_coord = np.array(self.robot_coord)
        ax.plot(self.robot_coord[:, 0], self.robot_coord[:, 1], 'r-', label="Vehicle Trajectory")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')
        ax.grid(False)
        ax.legend()
        plt.show(block=True)
