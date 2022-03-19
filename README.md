# Particle-Filter-SLAM
Implemented simultaneous localization and mapping (SLAM) using odometry, 2-D LiDAR scans, and stereo camera measurements from an autonomous car. Check the Samples folder for the output.

Details on the underlying technical approach can be found [here](https://drive.google.com/file/d/15kYTpraH4Hhz9RK780Gplf1EBF75bfpc/view?usp=sharing)
### Installation

This code uses Python 3.6.

- Install dependencies
```bash
$ pip install -r requirements.txt
```

### Data

- Download [Sensor amd Stereo Data](https://drive.google.com/drive/folders/1T6JFwsjxlZSxDNBdAJ05HMVRogkIVONb?usp=sharing) 
- Extract files to ```data```.
- The contents of ```data``` should be the following:
```
param
sensor_data
stereo_images
```


*Note: all python calls below must be run from ```./``` i.e. home directory of the project*
### Execution

The following will run the program with default config.
```bash
$ python main.py 
```

If we want the dead reckoning we can run the following command:
```bash
$ python main.py --dead-reckoning=True
```
By default, the dead_reckoning is disabled.

If we want the texture map to be generated at the end we can run the following command:
```bash
$ python main.py --texture-map=True
```
By default, the texture-map is disabled.


We can also set the number of particles, by running the follwing command
```bash
$ python main.py --particles=10
```
By default, the particle filter is initialized with 20 particles.