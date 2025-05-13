import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import cv2

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
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

        with open(label_path, 'r') as f:
            self.labels = json.load(f)

        self.image_files = list(self.labels.keys())
        self.num_images = len(self.image_files)
        self.index = 0
        self.epoch = 0

        if self.shuffle: # initial shuffle of data
            np.random.shuffle(self.image_files)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        end = self.index + self.batch_size
        images = []
        labels = []

        if end > self.num_images:
            # bath exceeds dataset end --> wrap around
            # first get all images from current dataset "run": batch_0
            # then get the rest of the images from the beginning of the dataset: batch_1
            batch_files_0 = self.image_files[self.index:]
            batch_files_1 = self.image_files[:end - self.num_images]
            batch_files = batch_files_0 + batch_files_1
            # update class attributes
            self.epoch += 1
            self.index = end - self.num_images

            if self.shuffle: # shuffle the data for the next epoch
                np.random.shuffle(self.image_files)
        else:
            # batch is within dataset --> just get the images
            batch_files = self.image_files[self.index:end]
            self.index = end

            if self.shuffle:
                np.random.shuffle(self.image_files)

        for filename in batch_files:
            # load image
            img_path = os.path.join(self.file_path, f"{filename}.npy")
            img = np.load(img_path)
            # resize image to image_size
            img = cv2.resize(img, (self.image_size[1], self.image_size[0])) # (height, width) = (image_size[0], image_size[1])

            img = self.augment(img)

            images.append(img)
            labels.append(self.labels[filename])

        return np.array(images), np.array(labels) # why not np.array(images) and np.array(labels)?
      

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        # introduce randomness using np.random
        if self.rotation:
            angle = np.random.choice([0, 90, 180, 270])
            if angle != 0:
                # rotate image using https://www.geeksforgeeks.org/python-opencv-cv2-rotate-method/
                img = cv2.rotate(img, {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE
                }[angle])

        if self.mirroring:
            if np.random.rand() > 0.5:
                # flip image horizontally using https://www.geeksforgeeks.org/python-opencv-cv2-flip-method/
                img = cv2.flip(img, 1)
        
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict.get(x, "Unknown class")
    
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