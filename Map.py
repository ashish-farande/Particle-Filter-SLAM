import numpy as np
import matplotlib.pyplot as plt

from sensor_utils import *
from matplotlib import animation
from IPython.display import clear_output
import cv2

MAP_SIZE = 1000

class Map:
    def __init__(self,x_min=-MAP_SIZE, y_min =-MAP_SIZE-1000, x_max=MAP_SIZE+1000, y_max=MAP_SIZE):

        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.res = 4
        self.x_size = int(np.ceil((x_max - x_min) / self.res + 1))  # cells
        self.y_size = int(np.ceil((y_max - y_min) / self.res + 1))  # cells
        self.odds = np.zeros((self.x_size, self.y_size))
        self.map = np.zeros((self.x_size, self.y_size), dtype="uint8")
        self.robot_coord = []
        # self.fig =  plt.figure()
        # self.fig , self.ax = plt.subplots(1,1)
        # self.im = self.ax.imshow(self.map, cmap='hot', vmin=0, vmax=1, animated=True)
        # self.fig.canvas.draw()

        # self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.map_init, frames=self.x_size * self.y_size, interval=10)

    def map_init(self):
        self.im.set_data(self.map)

    def update_free(self, start_point, end_points):
        end_points[:,0] = np.ceil((end_points[:,0] - self.x_min) / self.res).astype(np.int16) - 1
        end_points[:,1] = np.ceil((end_points[:,1] - self.y_min) / self.res).astype(np.int16) - 1
        start_point[0] = np.ceil((start_point[0] - self.x_min) / self.res).astype(np.int16) - 1
        start_point[1] = np.ceil((start_point[1] - self.y_min) / self.res).astype(np.int16) - 1
        self.robot_coord.append(start_point)
        newf, _ = get_mapping(start_point, end_points)
        xis, yis = newf[:,0].astype(np.int16), newf[:,1].astype(np.int16)
        indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < self.x_size)), (yis < self.y_size))
        self.odds[xis[indGood], yis[indGood]] -= np.log(10)
        self.map[xis[indGood], yis[indGood]] = 1

    def update_map(self):
        # self.map = np.where(self.odds.T < 0, 1, 0)
        pass

    def display(self):
        # plt.imshow(self.map, cmap='hot')
        # plt.show()
        # self.im.set_data(self.map)
        # self.fig.canvas.draw()
        img = self.map.astype(np.uint8) * 255

        for point in self.robot_coord:
            cv2.circle(img, (int(point[1]), int(point[0])),1 ,0, -1)
        cv2.imshow('map', img)
        cv2.waitKey(10)
        cv2.destroyAllWindows()

    def animate(self):
        self.im.set_data(self.map)
        return self.im

