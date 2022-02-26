import cv2
import numpy as np

from sensor_utils import *
from pr2_utils import *

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
        # self.fig =  plt.figure()
        # self.fig , self.ax = plt.subplots(1,1)
        # self.im = self.ax.imshow(self.map, cmap='hot', vmin=0, vmax=1, animated=True)
        # self.fig.canvas.draw()

        # self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.map_init, frames=self.x_size * self.y_size, interval=10)

    def map_init(self):
        # self.im.set_data(self.map)
        pass

    def update_free(self, in_start_point, in_end_points):
        end_points = self.convert_to_map(in_end_points)
        start_point = self.convert_to_map(in_start_point[np.newaxis, :])[0,:]

        # Storing the location to draw the trajectory
        self.robot_coord.append(start_point)

        # Trace the lines of the scans
        free_pixels, occupied_pixels = get_mapping(start_point, end_points)
        xis, yis = free_pixels[:, 0], free_pixels[:, 1]
        ind_good = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < self.x_size)), (yis < self.y_size))

        # Update the log odds
        added_logs = self.odds[xis[ind_good], yis[ind_good]] - np.log(4)
        clipped_vals = np.clip(added_logs, LAMBDA_MIN, LAMBDA_MAX)
        self.odds[xis[ind_good], yis[ind_good]] = clipped_vals

        # Faster way to update the map
        # self.map[xis[ind_good], yis[ind_good]] = 0
        self.update_map(xis[ind_good], yis[ind_good])

        xis, yis = end_points[:, 0], end_points[:, 1]
        ind_good = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < self.x_size)), (yis < self.y_size))
        added_logs = self.odds[xis[ind_good], yis[ind_good]] + np.log(4)
        clipped_vals = np.clip(added_logs, LAMBDA_MIN, LAMBDA_MAX)
        self.odds[xis[ind_good], yis[ind_good]] = clipped_vals
        self.update_map(xis[ind_good], yis[ind_good])




    def update_map(self, xs, ys):
        # self.map = np.where(self.odds.T < 0, 1, 0)
        self.map[xs, ys] = np.where(self.odds[xs, ys] < 0, 0, 1)
        pass

    def display(self, particles=None):
        # plt.imshow(self.map, cmap='hot')
        # plt.show()
        # self.im.set_data(self.map)
        # self.fig.canvas.draw()
        # img = self.map.astype(np.uint8) * 255

        scaled = self.odds+abs(LAMBDA_MIN)
        img = (scaled * 255/(2*LAMBDA_MAX)).astype(np.uint8)

        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # if particles:
        #     for point in particles:
        #         cv2.circle(img, (int(point[1]), int(point[0])), 1, (0,255,0), -1)
        # else:
        #     for point in self.robot_coord:
        #         cv2.circle(img, (int(point[1]), int(point[0])), 1, (0,255,0), -1)

        cv2.imshow('Log odds', img)
        cv2.waitKey(10)
        cv2.destroyAllWindows()


    def animate(self):
        # self.im.set_data(self.map)
        # return self.im
        pass

    def map_correlation(self, in_start_point, in_end_points ):
        ## TODO: Implement correlation
        end_points = self.convert_to_map(in_end_points)
        # start_point = self.convert_to_map(in_start_point[np.newaxis, :])[0, :]
        # newf, end_line = get_mapping(start_point, end_points)
        xis, yis = end_points[:, 0].astype(np.int16), end_points[:, 1].astype(np.int16)
        ind_good = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < self.x_size)), (yis < self.y_size))
        val = np.sum(self.map[xis[ind_good], yis[ind_good]])
        return val

    def convert_to_map(self, points):
        new_points = np.zeros_like(points)
        new_points[:, 0] = np.ceil((points[:, 0] - self.x_min) / self.res).astype(np.int16) - 1
        new_points[:, 1] = np.ceil((points[:, 1] - self.y_min) / self.res).astype(np.int16) - 1
        return new_points.astype(int)

    def build_map(self):
        img = ((1/(1+np.exp(-self.odds)) )* 255).astype(np.uint8)

        cv2.imshow('map', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


