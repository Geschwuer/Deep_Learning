from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

def main():
    # Test Checker class
    # checker = Checker(tile_size=2, resolution= 20)
    # checker.draw()
    # checker.show()

    # # Test Circle class
    # circle = Circle(1000, 100, (20, 500))
    # circle.draw()
    # circle.show()

    # spectrum = Spectrum(1000)
    # spectrum.draw()
    # spectrum.show()

    # Test ImageGenerator class
    generator = ImageGenerator(file_path="./data/exercise_data", 
                               label_path="./data/Labels.json",
                               batch_size = 32, 
                               image_size=[100, 100, 3],
                               rotation=True,
                               mirroring=True, 
                               shuffle=True)
    
    for i in range(1):
        generator.show()
        #generator.next()
        print(f"Epch: {generator.epoch}, Batch: {i+1}")

if __name__ == "__main__":
    main()