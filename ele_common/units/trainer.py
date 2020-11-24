from paddle import fluid
import numpy as np

class Trainer:
    def __init__(self,
                 name,
                 loss_function=fluid.layers.softmax_with_cross_entropy,
                 Optimizer=fluid.optimizer.AdamOptimizer,
                 learning_rate=0.01,
                ):
        self.Optimizer = Optimizer
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.train_losses = []

        self._mode = None
        self._optimizer = None
        self._name = name

    def train(self,
              model,
              reader,
              epoch_num=3,
              mode='train'
              ):
        # save model
        self._model = model
        with fluid.dygraph.guard():
            model.train()
            optimizer = self.Optimizer(learning_rate=self.learning_rate, parameter_list=model.parameters())

            self._optimizer = optimizer

            for epoch in range(epoch_num):
                for batch, data in enumerate(reader.train()):
                    imgs, labels = data
                    imgs = fluid.dygraph.to_variable(imgs)
                    labels = fluid.dygraph.to_variable(labels)
                    logits = model(imgs)
                    loss = self.loss_function(logits, labels)
                    avg_loss = fluid.layers.mean(loss)

                    if mode == 'debug':
                        print("label:", labels.numpy())
                        print("logits:", logits.numpy())
                        print("loss:", loss.numpy())


                    avg_loss.backward()
                    optimizer.minimize(avg_loss)
                    model.clear_gradients()

                    if batch % 100 == 0:
                        self.train_losses.append(avg_loss.numpy())
                        print(f"epoch:{epoch} batch:{batch} loss:{avg_loss.numpy()}")

                model.eval()
                losses = []
                accuracies = []
                for batch, data in enumerate(reader.test()):
                    imgs, labels = data
                    imgs = fluid.dygraph.to_variable(imgs)
                    labels = fluid.dygraph.to_variable(labels)
                    logits = model(imgs)
                    loss = self.loss_function(logits, labels)
                    accuracy = fluid.layers.accuracy(logits, labels)
                    accuracies.append(accuracy.numpy())
                    losses.append(loss.numpy())

                print(f"epoch:{epoch} test_result: accuracy/loss | {np.mean(accuracies)}/{np.mean(losses)}")
                model.train()

    def draw(self, *kwargs):
        import matplotlib.pyplot as plt
        x = np.linspace(1, len(self.train_losses), len(self.train_losses))
        y = self.train_losses
        plt.xlabel('batch')
        plt.ylabel('loss')
        plt.plot(x, y)

        plt.show()

    def save_state(self, path=None):
        import os

        if path is None:
            path = ''

        if self._model is None:
            raise KeyError('Model don\'t exist!')

        with fluid.dygraph.guard():
            fluid.save_dygraph(self._model, os.path.join(path, self._name))
            fluid.save_dygraph(self._optimizer, os.path.join(path, self._name))

    def load_state(self):
        # TODO: 读取参数
        pass
