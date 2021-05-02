import matplotlib.pyplot as plt
import numpy as np
from perlin_noise import PerlinNoise
from opensimplex import OpenSimplex
from scipy import interpolate as interpolation
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')


class TerrainGenerator:
    """This class generates terrain in 3D
    """

    def __init__(self, dimension=(5, 5), roughness=0.5, algorithm="perlin", seed=0):
        """Intializes the Terrain generator

        Args:
            dimension (tuple, optional): Dimension of the surface (length x width). Defaults to (5, 5).
            roughness (float, optional): surface roughness. Defaults to 0.5.
            algorithm (str, optional): Choice of algorithm 1. perlin 2. open-simplex. Defaults to "perlin".
            seed (int, optional): User defined seed value for generating random numbers. Defaults to 0.
        """
        try:
            # verify the the arg
            if not isinstance(dimension, set):
                raise TypeError(
                    "dimension need to be set. Dimension represents 2D surface of (length,width)")

            self.perlin_noise = PerlinNoise(octaves=roughness)
            self.open_simplex_noise = OpenSimplex(seed)
            self.dimension = dimension
            self.x_axis, self.y_axis = np.meshgrid(np.linspace(
                0, dimension[0], dimension[0]), np.linspace(0, dimension[1], dimension[1]))
            self.z_axis = np.full(shape=dimension, fill_value=1.0, dtype=float)
            self.height_function = None
            self.algorithm = algorithm
        except TypeError as e:
            logging.critical(e)
            exit(-1)
        except Exception as e:
            logging.critical(e)
            exit(-1)

    def generate(self):
        """Generates Terrain in 3D
        """
        try:
            if self.algorithm == "perlin":
                # 2D array to store pernil-noise
                perlin_noise_value_buffer2D = []
                # Generate perlin-noise
                for i in range(self.dimension[0]):
                    row = []
                    for j in range(self.dimension[1]):
                        noise_val = self.perlin_noise(
                            [i / self.dimension[0], j / self.dimension[1]])
                        row.append(noise_val)
                    perlin_noise_value_buffer2D.append(row)

                # Convert perlin noise value to surface height
                surface_height = perlin_noise_value_buffer2D
            elif self.algorithm == "open-simplex":
                # 2D array to store pernil-noise
                open_simple_noise_value_buffer2D = []
                # Generate perlin-noise
                for i in range(self.dimension[0]):
                    row = []
                    for j in range(self.dimension[1]):
                        noise_val = self.open_simplex_noise.noise2d(i, j)
                        row.append(noise_val)
                    open_simple_noise_value_buffer2D.append(row)
                # Convert perlin noise value to surface height
                surface_height = open_simple_noise_value_buffer2D
            else:
                raise RuntimeError("Algorithm is not supported")
            self.z_axis = np.array([np.array(xi) for xi in surface_height])
            self.height_function = interpolation.RectBivariateSpline(np.linspace(
                0, self.dimension[0], self.dimension[0]), np.linspace(0, self.dimension[1], self.dimension[1]), self.z_axis)
        except RuntimeError as e:
            logging.critical(e)
            exit(-1)
        except Exception as e:
            logging.critical(e)
            exit(-1)

    def get_height(self, x, y):
        """gets height of the surface 

        Args:
            x (float): x-axis dimension of the surface
            y (float): y-axis dimension of the surface

        Returns:
            float: height of the terrain
        """
        if self.height_function is not None:
            height_val = self.height_function(x, y)
            return np.float(height_val[0][0])
        else:
            return 0


if __name__ == "__main__":
    area_dimension = (100, 100)
    surface_roughness = 0.5

    surface_gen = TerrainGenerator(
        dimension=area_dimension, roughness=surface_roughness)
    surface_gen.generate()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(surface_gen.x_axis, surface_gen.y_axis,
                      surface_gen.z_axis, color='black')
    ax.set_title('wireframe')
    plt.show()

    print(surface_gen.get_height(0, 1))
