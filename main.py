from ParticleFilter import *

pf = ParticleFilter()

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

print("Done")
