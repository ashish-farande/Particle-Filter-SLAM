from ParticleFilter import *
from Sensor import *

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(PROJECT_PATH, 'data')
SENSOR_DATA_PATH = os.path.join(DATA_PATH, 'sensor_data')
PARAM_PATH = os.path.join(DATA_PATH, 'param')

LIDAR_DATA_FILE = os.path.join(SENSOR_DATA_PATH, 'lidar.csv')
FOG_DATA_FILE = os.path.join(SENSOR_DATA_PATH, 'fog.csv')
ENCODER_DATA_FILE = os.path.join(SENSOR_DATA_PATH, 'encoder.csv')

LIDAR_TO_VEHICLE_PARAMETERS_PATH = os.path.join(PARAM_PATH, 'Vehicle2Lidar.txt')
FOG_TO_VEHICLE_PARAMETERS_PATH = os.path.join(PARAM_PATH, 'Vehicle2FOG.txt')
STEREO_TO_VEHICLE_PARAMETERS_PATH = os.path.join(PARAM_PATH, 'Vehicle2Stereo.txt')

LEFT_CAMERA_CONFIG_FILE_PATH = os.path.join(PARAM_PATH, "left_camera.yaml")
RIGHT_CAMERA_CONFIG_FILE_PATH = os.path.join(PARAM_PATH, "right_camera.yaml")

# Create Sensors
lidar = Lidar(param_file=LIDAR_TO_VEHICLE_PARAMETERS_PATH)
drive = Drive(param_file=FOG_TO_VEHICLE_PARAMETERS_PATH)

# Set Up for the sensors
lidar.load_data(LIDAR_DATA_FILE)
drive.load_data(gyro_path=FOG_DATA_FILE, encoder_data=ENCODER_DATA_FILE)

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
