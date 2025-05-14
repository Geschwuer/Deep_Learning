from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator


def main():
    # Test Checker class
    # checker = Checker(tile_size=2, resolution= 20)
    # checker.draw()
    # checker.show()

    # circle = Circle(1000, 100, (20, 500))
    # circle.draw()
    # circle.show()

    # spectrum = Spectrum(resolution=100)
    # spectrum.draw()
    # spectrum.show()

    image_generator = ImageGenerator(
        file_path="c:\\FAU_Programming\\Deep_Learning\\exercise0\\src_to_implement\\exercise_data",
        label_path="C:\\FAU_Programming\\Deep_Learning\\exercise0\\src_to_implement\\Labels.json",
        batch_size=10,
        image_size=[50,50,3],
    )

    
    image_generator.next()
    image_generator.class_name(0)
    image_generator.show()

if __name__ == "__main__":
    main()