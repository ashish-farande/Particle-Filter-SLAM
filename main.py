from ParticleFilter import *
from Sensor import *
from Stereo import Stereo

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(PROJECT_PATH, 'data')
SENSOR_DATA_PATH = os.path.join(DATA_PATH, 'sensor_data')
IMAGES_DATA_PATH = os.path.join(DATA_PATH, 'stereo_images')
PARAM_PATH = os.path.join(DATA_PATH, 'param')

LIDAR_DATA_FILE = os.path.join(SENSOR_DATA_PATH, 'lidar.csv')
FOG_DATA_FILE = os.path.join(SENSOR_DATA_PATH, 'fog.csv')
ENCODER_DATA_FILE = os.path.join(SENSOR_DATA_PATH, 'encoder.csv')
LEFT_CAMERA_DATA_PATH = os.path.join(IMAGES_DATA_PATH, 'stereo_left')
RIGHT_CAMERA_DATA_PATH = os.path.join(IMAGES_DATA_PATH, 'stereo_right')


LIDAR_TO_VEHICLE_PARAMETERS_PATH = os.path.join(PARAM_PATH, 'Vehicle2Lidar.txt')
FOG_TO_VEHICLE_PARAMETERS_PATH = os.path.join(PARAM_PATH, 'Vehicle2FOG.txt')
STEREO_TO_VEHICLE_PARAMETERS_PATH = os.path.join(PARAM_PATH, 'Vehicle2Stereo.txt')

LEFT_CAMERA_CONFIG_FILE_PATH = os.path.join(PARAM_PATH, "left_camera.yaml")
RIGHT_CAMERA_CONFIG_FILE_PATH = os.path.join(PARAM_PATH, "right_camera.yaml")

# Create Sensors
lidar = Lidar(param_file=LIDAR_TO_VEHICLE_PARAMETERS_PATH)
drive = Drive(param_file=FOG_TO_VEHICLE_PARAMETERS_PATH)
stereo = Stereo(LEFT_CAMERA_CONFIG_FILE_PATH, RIGHT_CAMERA_CONFIG_FILE_PATH, STEREO_TO_VEHICLE_PARAMETERS_PATH)

# Set Up for the sensors
lidar.load_data(LIDAR_DATA_FILE)
drive.load_data(gyro_path=FOG_DATA_FILE, encoder_data=ENCODER_DATA_FILE)
stereo.load_data(LEFT_CAMERA_DATA_PATH, RIGHT_CAMERA_DATA_PATH)

# Create Particle Filter
pf = ParticleFilter(lidar_sensor=lidar, motion_sensor=drive)

# Initialise Map
pf.initialise_map()
# pf.show_map()

move_ts = drive.get_next_timestamp()
lidar_ts = lidar.get_next_timestamp()
stereo_ts = stereo.get_next_timestamp()


i = 0
while move_ts and lidar_ts:
    pf.predict()
    pf.update()
    # while stereo_ts and lidar_ts  and lidar_ts > stereo_ts:
    #     coord, pixel = stereo.read_sample()
    #     stereo_ts = stereo.get_next_timestamp()
    #     pf.texture_map(coord, pixel)
    pf.resample()

    if i % 100 == 0:
        pf.show_map()

    move_ts = drive.get_next_timestamp()
    lidar_ts = lidar.get_next_timestamp()
    i += 1

pf.map.build_map()
print("Done")