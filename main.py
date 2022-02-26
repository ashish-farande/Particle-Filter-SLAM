from ParticleFilter import *
from Sensor import *


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(PROJECT_PATH, 'data')
SENSOR_DATA_PATH = os.path.join(DATA_PATH, 'sensor_data')
PARAM_PATH = os.path.join(DATA_PATH, 'param')

LIDAR_DATA_FILE = 'lidar.csv'
FOG_DATA_FILE = 'fog.csv'
ENCODER_DATA_FILE = 'encoder.csv'

VEHICLE_TO_LIDAR = 'Vehicle2Lidar.txt'
VEHICLE_TO_FOG = 'Vehicle2FOG.txt'
VEHICLE_TO_STEREO = 'Vehicle2Stereo.txt'

LEFT_CAMERA_CONFIG = "left_camera.yaml"
RIGHT_CAMERA_CONFIG = "right_camera.yaml"

# Create Sensors
lidar = Lidar(param_file=os.path.join(PARAM_PATH, VEHICLE_TO_LIDAR))
drive = Drive(param_file=os.path.join(PARAM_PATH, VEHICLE_TO_FOG))

# Set Up for the sensors
lidar.load_data(os.path.join(SENSOR_DATA_PATH, LIDAR_DATA_FILE))
drive.load_data(gyro_path=os.path.join(SENSOR_DATA_PATH, FOG_DATA_FILE), encoder_data=os.path.join(SENSOR_DATA_PATH, ENCODER_DATA_FILE))

# Create Particle Filter
pf = ParticleFilter(lidar_sensor=lidar, motion_sensor=drive)

# Initialise Map
pf.initialise_map()
pf.show_map()


move_ts = pf.drive.get_next_timestamp()
lidar_ts = pf.lidar.get_next_timestamp()

i = 0
while move_ts and lidar_ts:
    pf.predict()
    pf.update()
    pf.resample()

    if i % 100 == 0:
        pf.show_map()

    move_ts = pf.drive.get_next_timestamp()
    lidar_ts = pf.lidar.get_next_timestamp()
    i += 1

pf.map.build_map()
print("Done")
