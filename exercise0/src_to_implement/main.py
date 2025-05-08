from pattern import Checker, Circle



def main():
    # Test Checker class
    checker = Checker(tile_size=2, resolution= 20)
    checker.draw()
    checker.show()

    circle = Circle(1000, 100, (20, 500))
    circle.draw()
    circle.show()

if __name__ == "__main__":
    main()