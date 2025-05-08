from pattern import Checker



def main():
    # Test Checker class
    checker = Checker(tile_size=2, resolution= 20)
    checker.draw()
    checker.show()

    # circle = Circle(100, 10, (50, 50))
    # circle.draw()
    # circle.show()

if __name__ == "__main__":
    main()