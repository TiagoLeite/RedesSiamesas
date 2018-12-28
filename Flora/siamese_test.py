from tensorflow.examples.tutorials.mnist import input_data
from siamese import Siamese
from pair import Pair
import os
import glob
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from random import shuffle

siamese = Siamese()
siamese.load_model()

all_pairs = list()

files1 = os.listdir('train_images/thunbergia_alata/')

for file1 in files1:
    pair_count = 0
    for file2 in files1[files1.index(file1) + 1:]:
        image = np.array(Image.open('train_images/lantana_camara/' + file1))
        # print(np.shape(image))
        embed = siamese.test_model([image])
        # print(embed)
        image = np.array(Image.open('train_images/thunbergia_alata/' + file2))
        # print(np.shape(image))
        embed2 = siamese.test_model([image])
        # print(embed2)
        diff = np.sum((embed - embed2) ** 2, axis=1)
        print(diff)

