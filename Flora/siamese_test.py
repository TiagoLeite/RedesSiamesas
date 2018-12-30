from tensorflow.examples.tutorials.mnist import input_data
from siamese import Siamese
from pair import Pair
import os
import glob
import pandas as pd
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from random import shuffle


def save_embed():
    file_to_save = pd.DataFrame(columns=('classId', 'className', 'embed'))
    folders = glob.glob('train_images/*')
    folders = sorted(folders)
    folders_dict = {k: folders[k].split('/')[1] for k in range(len(folders))}
    print(folders)
    print(folders_dict)
    print(folders_dict[3])
    row_count = 0
    for k in range(len(folders_dict)):
        files = os.listdir('train_images/' + folders_dict[k])
        files = sorted(files)
        # print(files1)
        for file1 in files:
            print(file1, folders_dict[k], k)
            image = np.array(Image.open('train_images/' + folders_dict[k] + '/' + file1))
            embed = siamese.test_model([image])[0]
            embed_str = ' '.join(str(embed[p]) for p in range(len(embed)))
            file_to_save.loc[row_count] = [k, folders_dict[k], embed_str]
            row_count += 1
    file_to_save.to_csv('embed.csv', index=False)


def row_mean(file, column_name):
    summary = file[file['className'] == column_name]
    embeds = []
    for index, row in summary.iterrows():
        values = [float(k) for k in str(row['embed']).split(' ')]
        embeds.append(values)
    means = np.mean(embeds, axis=0)
    return means


file = pd.read_csv('embed.csv')
# print(file['className'].unique())

means = []

for column_name in file['className'].unique():
    means.append(row_mean(file, column_name))

print(np.shape(means))

# for k in range(len(means)-1):
#    for j in range(k, len(means)):
#        dist = np.sqrt(np.sum((means[k] - means[j])**2))
#        print(k, j, dist)

siamese = Siamese()
siamese.load_model()

image = [np.array(Image.open('image_test.jpg')) / 255.0]
print(np.shape(image))
pred = siamese.test_model(image)

dists = []
for k in range(len(means)):
    dist = np.sqrt(np.sum((means[k] - pred) ** 2))
    print(k, dist)
    dists.append(dist)

print('Class:', np.argmin(dists))


