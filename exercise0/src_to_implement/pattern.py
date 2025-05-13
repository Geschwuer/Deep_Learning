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
    # 0 = black, 1=white
    def __init__(self, resolution, radius, position):
        self.output = None
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        # create x and y values for position as center point of the circle
        x0, y0 = self.position
        radius = self.radius

        # create meshgrid to function as coordinate system
        X, Y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))

        # pythagoras law, calculates the distance
        distance = np.sqrt((X-x0)**2 + (Y-y0)**2)

        # if point is inside radius --> 1, otherwise 0
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


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
    # image is a 2D array of shape (resolution, resolution, 3)
    # RGB values are in the range [0, 1]
        ls = np.linspace(0.0, 1.0, self.resolution)
        x, y = np.meshgrid(ls, ls)

        # x increases from left to right --> red
        # y increases from top to bottom --> green
        # blue increases from right to left --> inverse red
        R = x
        B = 1-x
        G = y

        # stack matrices on top of each other
        spectrum = np.stack((R, G, B), axis=2) # shape (resolution, resolution, 3)

        self.output = spectrum

        return self.output.copy()

    def show(self):
        if self.output is None:
            raise ValueError("You need to call draw() before show()")
        
        plt.imshow(self.output, cmap = "gray")
        plt.axis('off')
        plt.title(f"Binary Circle")
        plt.show()