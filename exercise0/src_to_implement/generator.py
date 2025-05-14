import os.path
import json
import scipy.misc
from skimage.transform import resize
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path : str, label_path:str, batch_size:int, image_size:list, rotation:bool=False, mirroring:bool=False, shuffle:bool=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size

        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        with open(self.label_path) as labels_file:
            self.labels = json.load(labels_file)

        # Load file names and paths
        file_names = [os.path.splitext(f)[0] for f in os.listdir(self.file_path)]
        total_paths = [os.path.join(self.file_path, f"{name}.npy") for name in file_names]

        # Shuffle at the start if needed
        if self.shuffle:
            combined = list(zip(total_paths, file_names))
            random.shuffle(combined)
            total_paths, file_names = zip(*combined)
            total_paths, file_names = list(total_paths), list(file_names)


        self.total_paths = total_paths
        self.file_names = file_names
        self.dataset_size = len(self.file_names)
        self.current_index = 0

        self.epoch = 0

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        # resize image
        


        if self.current_index >= self.dataset_size:
            self.current_index = 0
            self.epoch += 1
            if self.shuffle:
                combined = list(zip(self.total_paths, self.file_names))
                random.shuffle(combined)
                self.total_paths, self.file_names = zip(*combined)
                self.total_paths = list(self.total_paths)
                self.file_names = list(self.file_names)

        end_index = min(self.current_index + self.batch_size, self.dataset_size)
        batch_paths = self.total_paths[self.current_index:end_index]
        batch_names = self.file_names[self.current_index:end_index]

        self.current_index = end_index

        images_batch = []
        labels_batch = []

        for path, name in zip(batch_paths, batch_names):
            image = np.load(path)
            image_resized = resize(image, self.image_size)

            # mirror images randomly
            if self.mirroring:
                if random.random() > 0.5:
                    image_resized = image_resized[::-1, ::-1, :]
            # else:
            #     image_resized_mirrored = image_resized[::-1, :, :]
            if self.rotation:
                angle = random.choice([90, 180, 270])
                if angle == 90:
                    self.image = np.rot90(image_resized, k = 1)
                elif angle == 180:
                    self.image = np.rot90(image_resized, k = 2)
                elif angle == 270:
                    self.image = np.rot90(image_resized, k = 3)
            images_batch.append(image_resized)
            labels_batch.append(self.labels[name])

        return np.array(images_batch), np.array(labels_batch)


    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        return img

    def current_epoch(self):
        # return the current epoch number

        return self.epoch

    def class_name(self, x:int):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()

        cols = min(self.batch_size, 7) # max 7 images per row
        rows = int(np.ceil(self.batch_size / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
        axes = axes.flatten() # flatten the axes array to make it easier to iterate over

        for i in range(self.batch_size):
            ax = axes[i]
            ax.imshow(images[i], cmap = 'viridis')
            ax.set_title(self.class_name(labels[i]))
            ax.axis('off')

        # turn off remaining axes
        for j in range(self.batch_size, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

