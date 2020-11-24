ELE PROJECT
===========
    本项目为中央子项目 "基于深度学习的电磁数据拟合" 的开源代码库
    运行代码时，请先将 .dat 数据存放到同
    目录下的 data 文件夹中。

##  TODO LIST
    1. 增加 API 说明文件。
    2. 增加 项目更新日志。
    3. 增加 common.unit 等文件中的代码注释和描述。  
      
    4. 构建学习对数拟合参数的神经网络。
    5. *解决时间通道数不同的问题*。
    
##  Progress  
### 0. 总体进展
1. 已经完成读取输入的模块、类等
2. 已经完成将一个点的数据进行对数函数拟合并输出图像的函数
3. ~~已经完成神经网络初步搭建~~。
4. 完成搭建DrakNet53网络结构。 
### 1. 文件读取与处理 
在common.unit模块中，定义了SingleFile/SinglePoint类，分别用于读取单个
数据文件/从文件流中读取一个点，具体的实现见API。  
__读取一个点的示例__:  
```python
from ele_common.units import SingleFile

singlefile = SingleFile(filepath='../../data/origin/before/LINE_120_dbdt.dat')
print(singlefile._date)
print(singlefile.filename)
point = singlefile.get_one_point()

point.plot()
```
为了更好的进行数据分析，先使用 `pic_generator.py` 模块中的代码生成所有的数据图片。
```python
import os

import matplotlib.pyplot as plt

from ele_common.units import SingleFile

if __name__ == '__main__':
    father_dir = os.path.dirname(__file__)

    before_data_dir = os.path.join(father_dir, 'data/origin/before')
    after_data_dir = os.path.join(father_dir, 'data/origin/after')
    batch_size = 10

    figures_dir = os.path.join(os.path.join(father_dir, 'data'), 'figures')
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    for filename in os.listdir(before_data_dir):
        datafile = SingleFile(filepath=os.path.join(before_data_dir, filename))
        data_reader = datafile.get_reader(batch_size=batch_size)

        single_file_path = os.path.join(figures_dir, f'{datafile.filename}')

        if not os.path.exists(single_file_path):
            os.mkdir(single_file_path)

        for batch_id, points in enumerate(data_reader()):
            for point_id, point in enumerate(points):
                point_id = batch_id * batch_size + point_id
                print(f'[INFO] drawing {filename}--point_{point_id}...')
                plt.figure()
                point.plot(show=False)
                plt.savefig(
                    os.path.join(single_file_path, f'point_{point_id}.jpg')
                )
                plt.close()
```
### 2. 用对数函数进行图像拟合
具体函数在`solutions.polyfit.fit_one_point`中。  
该函数接受一个`SinglePoint`参数，并使用函数  

<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;a&space;\log(|bx&space;&plus;&space;ep|)&space;&plus;&space;c\log(|dx&plus;ep|)^2&space;&plus;&space;e\log(|fx&plus;ep|)^3&space;&plus;&space;g\log(|hx|&plus;ep)^4&space;&plus;&space;i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;a&space;\log(|bx&space;&plus;&space;ep|)&space;&plus;&space;c\log(|dx&plus;ep|)^2&space;&plus;&space;e\log(|fx&plus;ep|)^3&space;&plus;&space;g\log(|hx|&plus;ep)^4&space;&plus;&space;i" title="f(x) = a \log(|bx + ep|) + c\log(|dx+ep|)^2 + e\log(|fx+ep|)^3 + g\log(|hx|+ep)^4 + i" /></a>  

进行拟合。  
在圆滑后的数据中，可以得到较好的拟合结果。
```python
from ele_common.units import SingleFile
from solutions import fit_point

before_filepath = "data/origin/before/LINE_100_dbdt.dat"
after_filepath = "data/origin/after/new_LINE_100_dbdt.dat"

before_file = SingleFile(before_filepath)
after_file = SingleFile(after_filepath)

fit_point(before_file.get_one_point(), show=True)

```
对圆滑后的数据拟合如下:  

![](https://github.com/Sumbrella/ele_project/raw/master/else/pic/after_fit_example.png)  

对圆滑前的数据拟合如下:  

![](https://github.com/Sumbrella/ele_project/raw/master/else/pic/before_fit_example.png)  

**从上面的结果可以看出，直接拟合对圆滑后的图像拟合结果较好，对于圆滑前的图像具有一定的拟合能力，但是并不能满足要求。**

### 3. 神经网络的建立
综上，对于圆滑后的图像，使用对数函数拟合有较好的效果，因此，目前的想法是构建神经网络，输入一个点的数据，输出拟合函数的各项参数。
以此建立神经网络:
```python
import matplotlib.pyplot as plt
import pandas as pd

from paddle import fluid
from paddle.fluid import ParamAttr, Pool2D, Conv2D, Linear, BatchNorm
from paddle.fluid.regularizer import L2Decay

from solutions.polyfit.get_reader import get_reader
from solutions.polyfit.teacher_change import change_teacher


class ConvBNLayer(fluid.dygraph.Layer):
    """
    卷积 + 批归一化，BN层之后激活函数默认用leaky_relu
    """
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size=(1, 3),
                 stride=1,
                 groups=1,
                 padding=(0, 1),
                 act="leaky",
                 is_test=True):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0, 0.2)),
            bias_attr=False,
            act=None)

        self.batch_norm = BatchNorm(
            num_channels=num_filters,
            is_test=is_test,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0, 0.2),
                regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(100),
                regularizer=L2Decay(0.)))

        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

class AlexNet(fluid.dygraph.Layer):
    def __init__(self, num_classes=9):
        super(AlexNet, self).__init__()

        self.conv1 = ConvBNLayer(num_channels=2, num_filters=96, filter_size=11, stride=4, padding=5, act='leaky_relu')
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = ConvBNLayer(num_channels=96, num_filters=256, filter_size=5, stride=1, padding=2, act='relu')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv3 = ConvBNLayer(num_channels=256, num_filters=384, filter_size=3, stride=1, padding=1, act='leaky_relu')
        self.conv4 = ConvBNLayer(num_channels=384, num_filters=384, filter_size=3, stride=1, padding=1, act='leaky_relu')
        self.conv5 = ConvBNLayer(num_channels=384, num_filters=256, filter_size=3, stride=1, padding=1, act='leaky_relu')
        self.pool5 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

        self.fc1 = Linear(input_dim=256 * 3, output_dim=256 * 2, act='relu')
        self.drop_ratio1 = 0.5
        self.fc2 = Linear(input_dim=256 * 2, output_dim=256 * 1, act='relu')
        self.drop_ratio2 = 0.5
        self.fc3 = Linear(input_dim=256, output_dim=num_classes)

        self.conv_layers = [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.conv4, self.conv5,
                            self.pool5]

    def forward(self, x):
        for layers in self.conv_layers:
            x = layers(x)

        # x = self.bn1(x)

        x = fluid.layers.reshape(x, [x.shape[0], -1])

        x = self.fc1(x)
        x = fluid.layers.dropout(x, self.drop_ratio1)

        x = self.fc2(x)
        x = fluid.layers.dropout(x, self.drop_ratio2)

        x = self.fc3(x)

        return x

```

### 4. 网络的训练
```python
    import numpy as np
    from ele_common.units import SingleFile, Reader

    train_dir = "../../data/train/before"
    train_label_dir = "../../data/train/teacher"
    test_dir = "../../data/test/before"
    test_label_dir = "../../data/test/teacher"

    mode = 'train'
    epoch_num = 150
    loss_function = fluid.layers.square_error_cost
    batch_size = 10
    bd = [1000, 2000, 3000]
    value = [1.0, 0.5, 0.1, 0.05]
    # lr = fluid.layers.piecewise_decay(boundaries=[1000, 2000, 3000], values=[1.0, 0.5, 0.1, 0.05])
    lr = 1.0
    l2 = fluid.regularizer.L2Decay(regularization_coeff=1e-2)

    train_reader = get_reader(train_dir, train_label_dir, batch_size=batch_size)
    test_reader = get_reader(test_dir, test_label_dir, batch_size=batch_size)
    min_losses = 1000000

    train_losses = []
    test_losses = []

    with fluid.dygraph.guard():
        model = AlexNet()
        reader = Reader(train=train_reader, test=test_reader)

        model.train()
        optimizer = fluid.optimizer.Adam(learning_rate=0.1,
                                         parameter_list=model.parameters(),
                                         # regularization=l2,
                                         )

        for epoch in range(epoch_num):
            for batch, data in enumerate(reader.train()):
                imgs, labels = data
                labels = change_teacher(labels)

                imgs = fluid.dygraph.to_variable(imgs)
                labels = fluid.dygraph.to_variable(labels)
                logits = model(imgs)
                loss = loss_function(logits, labels)
                avg_loss = fluid.layers.mean(loss)

                if batch % 10 == 0:
                    train_losses.append(avg_loss.numpy())
                    print(f"epoch:{epoch} batch:{batch} loss:{avg_loss.numpy()}")

                if mode == 'debug':
                    print("label:", labels.numpy())
                    print("logits:", logits.numpy())
                    print("loss:", loss.numpy())
                    print("avg_loss:", avg_loss.numpy())

                avg_loss.backward()

                optimizer.minimize(avg_loss)

                model.clear_gradients()

            model.eval()

            losses = []

            for batch, data in enumerate(reader.test()):
                imgs, labels = data

                # change labels
                labels = change_teacher(labels)

                imgs = fluid.dygraph.to_variable(imgs)

                labels = fluid.dygraph.to_variable(labels)

                logits = model(imgs)

                loss = loss_function(logits, labels)
                avg_loss = fluid.layers.mean(loss)
                losses.append(avg_loss.numpy())

            if np.mean(losses) < min_losses:
                min_losses = np.mean(losses)
                fluid.save_dygraph(model.state_dict(), "min_polyfit")
                fluid.save_dygraph(optimizer.state_dict(), "min_polyfit")

            print(f"epoch:{epoch} test_result: loss | {np.mean(losses)}")

            model.train()

        fluid.save_dygraph(model.state_dict(), "polyfit")
        fluid.save_dygraph(optimizer.state_dict(), "polyfit")

    train_losses = pd.DataFrame(train_losses)
    train_losses.plot()
    plt.show()
```

### 5. 网络的预测
```python
import os

from paddle import fluid
import numpy as np
import matplotlib.pyplot as plt

from solutions.polyfit.example import AlexNet
from solutions.polyfit.teacher_change import back_change
from ele_common.units import SingleFile
from ele_common.functions.polyfit import exponenial_func


def main():
    data_path = "../../data/train/before/LINE_100_dbdt.dat"

    with fluid.dygraph.guard():
        model = AlexNet()
        # Load static
        min_dict, _ = fluid.load_dygraph(model_path='min_polyfit')
        # print(min_dict)
        model.set_dict(stat_dict=min_dict)

        model.eval()

        data_file = SingleFile(data_path)
        one_point = data_file.get_one_point()

        data = one_point.get_data()

        data = np.array(data, 'float32').reshape(1, 2, 1, 100)
        # data.res
        data = fluid.dygraph.to_variable(data)

        logits = model(data)

        result = logits.numpy()

        result = back_change(result)

    x_data = one_point.x
    print("RESULT: \n", result)
    one_point.plot(show=False, label='origin')
    plt.plot(x_data, [exponenial_func(x, *result[0]) for x in x_data], label='predict')
    plt.show()


if __name__ == '__main__':
    main()

```
