import os

from pr2_utils import *
from Sensor import *
from Map import *
from sensor_utils import *
import copy

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(PROJECT_PATH, 'data')
SENSOR_DATA_PATH = os.path.join(DATA_PATH, 'sensor_data')

LIDAR_DATA_FILE = 'lidar.csv'
FOG_DATA_FILE = 'fog.csv'
ENCODER_DATA_FILE = 'encoder.csv'

LIDAR_ANGLES = np.linspace(-5, 185, 286) / 180 * np.pi



print(DATA_PATH)
lidar = Lidar()
lidar.load_data(os.path.join(SENSOR_DATA_PATH, LIDAR_DATA_FILE))

move = Move_Robot()
move.load_data(gyro_path=os.path.join(SENSOR_DATA_PATH, FOG_DATA_FILE), encoder_data=os.path.join(SENSOR_DATA_PATH, ENCODER_DATA_FILE))

global_map = Map()

robot_pose = np.array([0.0,0.0,0.0])
angles, ranges= get_valid(LIDAR_ANGLES, lidar.data[0])
coord = convert_angle_coord(angles, ranges)

d = lidar.convert_to_body_frame(coord)
laser_coord = convert_to_world_frame(robot_pose[2], robot_pose[0:2], d[:,:2])

global_map.update_free(copy.deepcopy(robot_pose[0:2]), laser_coord)
global_map.update_map()
global_map.display()

move_ts = move.get_next_timestamp()
lidar_ts = lidar.get_next_timestamp()
i=0


while move_ts and lidar_ts:

    # Predict
    d_robot_pose = np.array([0.0,0.0])

    while move_ts < lidar_ts:
        d_robot_pose += move.read_sample()
        move_ts = move.get_next_timestamp()

    robot_pose[0] += d_robot_pose[0]*np.cos(robot_pose[2])
    robot_pose[1] += d_robot_pose[0]*np.sin(robot_pose[2])
    robot_pose[2] += d_robot_pose[1]
    # Add this to the particles and add noise

    # Update part
    # 1. Map Correlation -> weigths of the particles


    lidar_data = lidar.read_sample()
    angles, ranges= get_valid(LIDAR_ANGLES, lidar_data)
    coord = convert_angle_coord(angles, ranges)
    S_B = lidar.convert_to_body_frame(coord)

    # 2. Select the particle with the highest weight
    laser_coord = convert_to_world_frame(robot_pose[2], robot_pose[0:2], S_B[:,:2])

    global_map.update_free(copy.deepcopy(robot_pose[0:2]), laser_coord)
    global_map.update_map()
    if i%100 == 0:
        # print(i, lidar_ts)
        global_map.display()
        # print(robot_pose)
    # move_ts = move.get_next_timestamp()
    lidar_ts = lidar.get_next_timestamp()
    i+=1
print("Done")


def handler():
    move_ts = move.get_next_timestamp()
    lidar_ts = lidar.get_next_timestamp()
    # while move_ts and lidar_ts:





