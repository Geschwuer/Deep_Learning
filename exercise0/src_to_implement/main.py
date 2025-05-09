from pattern import Checker, Circle, Spectrum



def main():
    # Test Checker class
    checker = Checker(tile_size=2, resolution= 20)
    checker.draw()
    checker.show()

    circle = Circle(1000, 100, (20, 500))
    circle.draw()
    circle.show()

    spectrum = Spectrum(1000)
    spectrum.draw()
    spectrum.show()

if __name__ == "__main__":
    main()