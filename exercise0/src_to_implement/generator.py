import os.path
import json
import scipy.misc
from skimage.transform import resize
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path : str, label_path:str, batch_size:str, image_size:list, rotation:bool=False, mirroring:bool=False, shuffle:bool=False):
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

        self.current_index = 0

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        # resize image
        


        with open(self.label_path) as labels_file:
            labels = json.load(labels_file)
        
        # list of all files
        file_paths = os.listdir(self.file_path)

        total_paths = [] # abosolute paths of all .npy files
        file_names = [] # list of all file names without file ending --> e.g. "3"

        for file in file_paths:
            total_paths.append(os.path.join(self.file_path, file))
            file_name = os.path.splitext(file)[0]
            file_names.append(file_name)


        for _ in range(self.batch_size):
            # if dataset is not evenly dividable by batch_size
            if self.current_index >= len(file_names):
                self.index = 0

            current_batch = total_paths[self.current_index:self.batch_size]  # paths for images in current batch
            current_batch_label = file_names[self.current_index:self.batch_size] # file names for labels in current batch (file names)

            self.current_index = self.current_index + self.batch_size

            images_batch = [] # all images in current batch
            labels_batch = [] # all labels in current batch
            for file_path, batch_label_index in zip(current_batch, current_batch_label):
                batch_file = np.load(file_path)

                # resize image after .npy file is loaded
                batch_file_resized = resize(batch_file, self.image_size)

                images_batch.append(batch_file_resized)
                labels_batch.append(labels[batch_label_index])

        return images_batch, labels_batch


    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        return img

    def current_epoch(self):
        # return the current epoch number
        return 0

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        pass

