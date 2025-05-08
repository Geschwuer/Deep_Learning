from pattern import Checker

def main():
    # Test Checker class
    checker = Checker(tile_size=2, resolution= 20)
    checker.draw()
    checker.show()

if __name__ == "__main__":
    main()