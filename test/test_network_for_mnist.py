from paddle import fluid

from common.units.reader import Reader
from common.units.trainer import Trainer
from solutions.polyfit.main import ConvBNLayer, DownSample, LayerWarp

elt_network_ctg = [2, 8, 4]


class EleNetwork(fluid.dygraph.Layer):
    def __init__(self, stages=None, is_test=True):
        super(EleNetwork, self).__init__()
        if stages is None:
            stages = elt_network_ctg

        self.stages = stages
        self.darknet53_conv_block_list = []
        self.downsample_list = []

        self.conv0 = ConvBNLayer(
            ch_in=1,
            ch_out=32,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test
        )

        self.downsample0 = DownSample(
            ch_in=32,
            ch_out=32*2,
            is_test=is_test,
            filter_size=3,
            padding=1,
        )

        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer(
                "stage_%d" % (i),
                LayerWarp(32 * (2 ** (i + 1)),
                          32 * (2 ** i),
                          stage,
                          is_test=is_test))
            self.darknet53_conv_block_list.append(conv_block)

        # 两个层级之间使用DownSample将尺寸减半
        for i in range(len(self.stages) - 1):
            downsample = self.add_sublayer(
                "stage_%d_downsample" % i,
                DownSample(ch_in=32 * (2 ** (i + 1)),
                           ch_out=32 * (2 ** (i + 2)),
                           is_test=is_test))
            self.downsample_list.append(downsample)

        self.fc1 = fluid.dygraph.Linear(input_dim=256*16, output_dim=10, act='relu')

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.downsample0(out)
        blocks = []
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list):  # 依次将各个层级作用在输入上面
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)
        out = fluid.layers.reshape(x=out, shape=[-1, 256*16])
        out = self.fc1(out)
        return out


if __name__ == '__main__':
    reader = Reader.from_paddle(data_name="mnist")
    train = Trainer(name="mnist",
                    # loss_function=fluid.layers.cross_entropy,
                    # Optimizer=fluid.optimizer.Adam,
                    )
    with fluid.dygraph.guard():
        model = EleNetwork()
        train.train(model, reader, epoch_num=2, mode='train')


