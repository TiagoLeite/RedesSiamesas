from tensorflow.examples.tutorials.mnist import input_data
from siamese import Siamese
from pair import Pair
import os
import glob
import random
import numpy as np

random.seed(1997)

EPISODE_MAX = 100000
BATCH_SIZE = 128


def get_all_pairs():
    folders = glob.glob('train/*')
    cont = 0
    pair_count = 0
    pairs = list()
    print(len(folders))
    for j in range(len(folders)):
        folder_cont = 0
        folder = folders[j]
        cont += 1
        files = os.listdir(folder)  # dir is your directory path
        number_files = len(files)
        # print(folder, cont, number_files)
        # folders.remove(folder)
        files = sorted(files)
        for k in range(len(files)):
            file = files[k]
            # print('Current:', file)
            # print('Size:', len(files[k:]))
            pair_count = 0
            for another_file in files[k + 1:]:
                pair_count += 1
                par = Pair(folder + '/' + file, folder + '/' + another_file, 1)
                # print(file, another_file)
                # par.print_shapes()
                folder_cont += 1
                pairs.append(par)
                if pair_count > 10:
                    break
        # print(len(pairs))
        cont_neg = 0
        for k in range(j, len(folders)):
            cont_neg += 1
            # if k % 1000 == 0:
            # print('K:', k)
            fold = folders[k]
            if fold == folder:
                # print(fold, folder)
                continue
            par = Pair(folder + '/1.jpg', fold + '/1.jpg', 0)
            folder_cont += 1
            pairs.append(par)
            if cont_neg >= 1088:
                break

        if cont_neg < 1000:
            for k in range(0, j):
                cont_neg += 1
                # if k % 100 == 0:
                # print('K:', k)
                fold = folders[k]
                if fold == folder:
                    # print(fold, folder)
                    continue
                par = Pair(folder + '/1.jpg', fold + '/1.jpg', 0)
                folder_cont += 1
                pairs.append(par)
                if cont_neg >= 1000:
                    break

        print(folder_cont, len(pairs))

        # if cont >= 1:
        #    break
    return pairs


def train_model(model, dataset):
    # Train model
    for episode in range(EPISODE_MAX):
        input_1, label_1 = dataset.train.next_batch(BATCH_SIZE)
        input_2, label_2 = dataset.train.next_batch(BATCH_SIZE)
        label = (label_1 == label_2).astype('float')

        # print(label)

        train_loss = model.train_model(input_1=input_1, input_2=input_2, label=label)

        if episode % 100 == 0:
            print('episode %d: train loss %.3f' % (episode, train_loss))

        if episode % 10000 == 0:
            model.save_model()


def test_model(model, dataset):
    # Test model
    embed = model.test_model(input_1=dataset.test.images)
    embed.tofile('embed.txt')


def main():
    # Load MNIST dataset
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    # Initialze model
    # siamese = Siamese()
    pairs = list()
    # for k in range(4):
    #    apar = Pair('aug_test/' + str((k % 2) + 1) + '.jpg', 'aug_test/' + str((k + 1) % 2 + 1) + '.jpg', k % 2)
    #    pairs.append(apar)
    # print(len(pairs))
    all_pairs = get_all_pairs()
    print('Pairs:', len(all_pairs))
    pos = [x for x in all_pairs if x.label == 1]
    print(len(pos))
    neg = [x for x in all_pairs if x.label == 0]
    print(len(neg))
    print((len(pos)+len(neg)))
    # for par in pairs:
    #    par.print_label()
    #    par.print_images()
    # train_model(model=siamese, dataset=mnist)
    # test_model(model=siamese, dataset=mnist)


if __name__ == '__main__':
    main()
