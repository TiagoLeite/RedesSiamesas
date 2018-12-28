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

EPISODE_MAX = 1000000
BATCH_SIZE = 64


def get_batch(all_pairs, start, end):
    input_1 = [np.array(Image.open(x.path_image_1).convert('RGB'))/255.0 for x in all_pairs[start:end]]
    input_2 = [np.array(Image.open(x.path_image_2).convert('RGB'))/255.0 for x in all_pairs[start:end]]
    labels = [x.label for x in all_pairs[start:end]]
    print(np.shape(input_1), np.shape(input_2), np.shape(labels))
    return input_1, input_2, labels


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
                if pair_count > 3:
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
            if cont_neg >= 500:
                break

        if cont_neg < 500:
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
                if cont_neg >= 500:
                    break

        print(folder_cont, len(pairs))

        # if cont >= 1:
        #    break
    return pairs


def train_model(model, all_pairs):

    for episode in range(EPISODE_MAX):
        input_1, input_2, labels = get_batch(all_pairs, episode*BATCH_SIZE, (episode+1)*BATCH_SIZE)
        train_loss = model.train_model(input_1=input_1, input_2=input_2, label=labels)
        #if episode % 2 == 0:
        print('episode %d: train loss %.5f' % (episode, train_loss))
        #if episode % 10000 == 0:
        #    model.save_model()


def test_model(model, dataset):
    # Test model
    embed = model.test_model(input_1=dataset.test.images)
    embed.tofile('embed.txt')


def main():

    siamese = Siamese()
    all_pairs = get_all_pairs()

    #print('Pairs:', len(all_pairs))
    #print(all_pairs[0].print_images())

    #for i in range(1, len(all_pairs)):
    #    index = random.randrange(0, i)
    #    all_pairs[index], all_pairs[i] = all_pairs[i], all_pairs[index]

    #print('Pairs:', len(all_pairs))
    #print(all_pairs[0].print_images())

    shuffle(all_pairs)

    '''for k in range(10):
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(a[k])
        axarr[1].imshow(b[k])
        if c[k] == 1:
            axarr[0].set_title('Same')
            axarr[1].set_title('Same')
        else:
            axarr[0].set_title('Different')
            axarr[1].set_title('Different')
        # print(a[0] + ' | ' + b[0])
        plt.show()'''

    pos = [x for x in all_pairs if x.label == 1]
    print('Pos:', len(pos))
    neg = [x for x in all_pairs if x.label == 0]
    print('Neg:', len(neg))
    print((len(pos)+len(neg)))

    # for par in all_pairs[:30]:
    #    par.print_label()
    #    par.print_images()

    train_model(siamese, all_pairs)
    # test_model(model=siamese, dataset=mnist)


if __name__ == '__main__':
    main()
