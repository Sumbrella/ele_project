import paddle
import numpy as np


class Reader:
    """
    this class has method "train" "test" which will return
    a teacher generator with batch_size(=10)
    """
    def __init__(self, train, test):
        self.train = train
        self.test = test

    @staticmethod
    def _reader(generator, batch_size, data_shape, label_shape=None):

        if label_shape is None:
            label_shape = [1]

        def reader():
            data = []
            labels = []
            for one_data, label in generator():
                data.append(np.reshape(one_data, data_shape).astype('float32'))
                labels.append(np.reshape(label, label_shape).astype('int64'))

                if len(data) == batch_size:
                    # teacher = np.array(imgs, 'float32').reshape(batch_size, 1, 28, 28)
                    # labels = np.array(labels, 'int32').reshape(len(labels), -1)
                    yield np.array(data), np.array(labels)
                    data = []
                    labels = []

            if len(data):
                yield np.array(data), np.array(labels)

        return reader

    @staticmethod
    def from_paddle(data_name='mnist', batch_size=10):

        train = None
        test = None
        data_shape = None
        label_shape = None

        if data_name == 'mnist':
            train = paddle.dataset.mnist.train
            test = paddle.dataset.mnist.test
            data_shape = [1, 28, 28]
            label_shape = [1]
        elif data_name == 'mq2007':
            train = paddle.dataset.mq2007.train
            test = paddle.dataset.mq2007.train
            data_shape = []
            label_shape = []
        elif data_name == 'xxx':
            # TODO: Add other packages
            pass

        else:
            raise ValueError('No package named {}'.format(data_name))

        return Reader(
                Reader._reader(train(), batch_size, data_shape, label_shape),
                Reader._reader(test(), batch_size, data_shape, label_shape)
             )