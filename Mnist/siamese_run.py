from tensorflow.examples.tutorials.mnist import input_data
from siamese_simple import Siamese
import numpy as np

EPISODE_MAX = 100
BATCH_SIZE = 100


def train_model(model, dataset):
    # Train model
    for episode in range(EPISODE_MAX):
        input_1, label_1 = dataset.train.next_batch(BATCH_SIZE)
        input_2, label_2 = dataset.train.next_batch(BATCH_SIZE)
        label = (label_1 == label_2).astype('float')

        #print(np.shape(input_1), np.shape(input_2), np.shape(label))
        #print(label)

        train_loss = model.train_model(input_1=input_1, input_2=input_2, label=label)

        if episode % 25 == 0:
            print('episode %d: train loss %.3f' % (episode, train_loss))

        if episode % 100 == 0:
            model.save_model()


def test_model(model, dataset):
    # Test model
    embed = model.test_model(input_1=dataset.test.images)
    embed.tofile('embed.txt')


def main():
    # Load MNIST dataset
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    # Initialze model
    siamese = Siamese()
    # Train model
    train_model(model=siamese, dataset=mnist)
    # Test model
    test_model(model=siamese, dataset=mnist)


if __name__ == '__main__':
    main()
