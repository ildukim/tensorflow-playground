import numpy as np

# from https://gist.github.com/ischlag/41d15424e7989b936c1609b53edd1390

def extract_data(filename, num_images):
    with open(filename, mode='rb') as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28 * 28)
        return data


def extract_labels(filename, num_images):
    with open(filename, mode='rb') as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
        one_hot_labels = np.zeros((num_images, 10))
        one_hot_labels[np.arange(num_images), labels] = 1
        return one_hot_labels


def load_mnist_data():
    train_images = extract_data('./data/train-images.idx3-ubyte', 60000)
    train_labels = extract_labels('./data/train-labels.idx1-ubyte', 60000)
    test_images = extract_data('./data/t10k-images.idx3-ubyte', 10000)
    test_labels = extract_labels('./data/t10k-labels.idx1-ubyte', 10000)
    return (train_images, train_labels, test_images, test_labels)
