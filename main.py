import argparse

from Vehicle import Vehicle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Particle Filter by Ashish Farande')
    parser.add_argument('--texture-map', type=bool, default=False,
                        help='Enable Texture mapping (default: False)')
    parser.add_argument('--dead-reckoning', type=bool, default=False,
                        help='Enable Dead Reckoning (default: False)')
    parser.add_argument('--particles', type=int, default=20,
                        help='Number of Particles (default: 20)')

    parameters = parser.parse_args()

    my_car = Vehicle(n_particles=parameters.particles, enable_texture_mapping=parameters.texture_map, enable_dead_reckoning=parameters.dead_reckoning)

    my_car.start()
    my_car.drive()

    my_car.show_map()
