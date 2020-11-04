ELE PROJECT
===========
    本项目为中央子项目 "基于深度学习的电磁数据拟合" 的开源代码库
    使用的电磁数据不作提供，运行代码时，请先将 .dat 数据存放到同
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
    3. 已经完成神经网络初步搭建。
### 1. 文件读取与处理 
在common.unit模块中，定义了SingleFile/SinglePoint类，分别用于读取单个
数据文件/从文件流中读取一个点，具体的实现见API。  
__读取一个点的示例__:  
```python
from common.unit import SingleFile

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

from common.unit import SingleFile

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
from common.unit import SingleFile
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
TODO:
