import numpy as np
import matplotlib.pyplot as plt

class Checker:
    # 1 = white, 0 = black
    def __init__(self, resolution, tile_size):
        assert resolution % (2*tile_size) == 0, \
            "Resolution must be divisible by 2*tile size"
        self.tile_size = tile_size
        self.resolution = resolution
        self.output = None

    def draw(self):
        base_pattern = np.array([[0, 1], 
                                 [1, 0]])
        
        # number of times to repeat the base pattern
        num_repeats = self.resolution // (2 * self.tile_size)

        # repeat base pattern
        pattern = np.tile(base_pattern, (num_repeats, num_repeats))
        
        # scale pattern to tile size
        mask = np.ones((self.tile_size, self.tile_size))
        pattern = np.kron(pattern, mask) # https://de.wikipedia.org/wiki/Kronecker-Produkt

        self.output = pattern

        return self.output.copy()

    def show(self):
        if self.output is None:
            raise ValueError("You need to call draw() before show()")
        
        plt.imshow(self.output, cmap = "gray")
        plt.axis('off')
        plt.title(f"Checkerboard Pattern: {self.tile_size}x{self.tile_size} tiles")
        plt.show()


class Circle:
    #0 = black, 1=white
    def __init__(self, resolution, radius, position):
        self.output = None
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        #create meshgrid from two arrays
        x0, y0 = self.position
        radius = self.radius

        Y, X = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))

        distance = np.sqrt((X-x0)**2 + (Y-y0)**2)

        mask = (distance <= radius).astype(int)

        self.output = mask

        return self.output.copy()

    def show(self):
        if self.output is None:
            raise ValueError("You need to call draw() before show()")
        
        plt.imshow(self.output, cmap = "gray")
        plt.axis('off')
        plt.title(f"Binary Circle")
        plt.show()

