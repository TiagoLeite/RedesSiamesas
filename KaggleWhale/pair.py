import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class Pair(object):

    def __init__(self, path_image_1, path_image_2, label):
        self.path_image_1 = path_image_1
        self.path_image_2 = path_image_2
        # self.image_1 = np.array(Image.open(path_image_1))
        # self.image_2 = np.array(Image.open(path_image_2))
        self.label = label

    def print_shapes(self):
        print(np.shape(self.image_1), np.shape(self.image_2), np.shape(self.label))

    def print_images(self):
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(self.image_1)
        axarr[1].imshow(self.image_2)
        if self.label == 1:
            axarr[0].set_title('Same')
            axarr[1].set_title('Same')
        else:
            axarr[0].set_title('Different')
            axarr[1].set_title('Different')
        print(self.path_image_1 + ' | ' + self.path_image_2)
        plt.show()

    def print_label(self):
        print(self.label)

    def get_label(self):
        return self.label