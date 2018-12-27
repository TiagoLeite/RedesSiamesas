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

# random.seed(1997)

BATCH_SIZE = 100
EPISODE_MAX = 5*int(15000/BATCH_SIZE)


def get_batch(all_pairs, start, end):
    input_1 = [np.array(Image.open(x.path_image_1)) / 255.0 for x in all_pairs[start:end]]
    input_2 = [np.array(Image.open(x.path_image_2)) / 255.0 for x in all_pairs[start:end]]
    labels = [x.label for x in all_pairs[start:end]]
    # print(np.shape(input_1), np.shape(input_2), np.shape(labels))
    return input_1, input_2, labels


def get_negative_pairs(all_folders, folder, all_pairs):
    files1 = os.listdir(folder)
    last_len = len(all_pairs)
    #shuffle(all_folders)
    for neg_folder in all_folders:
        if neg_folder == folder:
            #print('Skipping:', neg_folder, folder)
            continue
        files2 = os.listdir(neg_folder)
        shuffle(files1)
        shuffle(files2)
        for k in range(36):
            par = Pair(folder + '/' + files1[k], neg_folder + '/' + files2[k], 0)  # different class
            #par.print_images()
            all_pairs.append(par)

    print('NEG:', folder, len(all_pairs) - last_len)


def get_positive_pairs(folder, all_pairs):
    files1 = os.listdir(folder)
    shuffle(files1)
    last_len = len(all_pairs)
    files1 = files1[:32]
    for file1 in files1:
        pair_count = 0
        for file2 in files1[files1.index(file1) + 1:]:
            if file1 == file2:  # not necessary, actually
                continue
            pair_count += 1
            par = Pair(folder + '/' + file1, folder + '/' + file2, 1)  # Same class
            all_pairs.append(par)

    print('POS:', folder, len(all_pairs) - last_len)


def get_all_pairs():
    folders = glob.glob('train_images/*')
    pairs = list()
    print(len(folders))
    for j in range(len(folders)):
        folder = folders[j]
        get_positive_pairs(folder, pairs)
        get_negative_pairs(folders, folder, pairs)
    return pairs


def train_model(model, all_pairs):
    for episode in range(EPISODE_MAX-1):
        input_1, input_2, labels = get_batch(all_pairs, episode * BATCH_SIZE, (episode + 1) * BATCH_SIZE)
        train_loss = model.train_model(input_1=input_1, input_2=input_2, label=labels)
        # if episode % 2 == 0:
        print('episode %d: train loss %.5f' % (episode, train_loss))
        if episode % 10 == 9:
            print('Saving...')
            model.save_model()


def test_model(model, dataset):
    # Test model
    embed = model.test_model(input_1=dataset.test.images)
    embed.tofile('embed.txt')


def main():

    siamese = Siamese()
    all_pairs = get_all_pairs()
    print('Pairs:', len(all_pairs))
    shuffle(all_pairs)

    #for k in range(10):
    #    print(all_pairs[k].print_images())

    pos = [x for x in all_pairs if x.label == 1]
    print('Pos:', len(pos))
    neg = [x for x in all_pairs if x.label == 0]
    print('Neg:', len(neg))
    print((len(pos) + len(neg)))

    train_model(siamese, all_pairs)
    #test_model(model=siamese, dataset=mnist)


if __name__ == '__main__':
    main()
