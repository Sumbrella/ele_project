ELE PROJECT
===========
    本项目为中央子项目 "基于深度学习的电磁数据拟合" 的开源代码库
    使用的电磁数据不作提供，运行代码时，请先将 .dat 数据存放到同
    目录下的 data 文件夹中。

##  TODO LIST
    1. 增加 API 说明文件。
    2. 增加 项目更新日志。
    3. 增加 common.unit 等文件中的代码注释和描述。
  
##  Progress  
### 1. 文件读取与处理 
在common.unit模块中，定义了SingleFile/SinglePoint类，分别用于读取单个
数据文件/从文件流中读取一个点，具体的实现见API。  
__示例__:  
```python
from common.unit import SingleFile

singlefile = SingleFile(filepath='../../data/origin/before/LINE_120_dbdt.dat')
print(singlefile._date)
print(singlefile.filename)
point = singlefile.get_one_point()

point.plot()
```

### 2. 用对数函数进行图像拟合
具体函数在`solutions.polyfit.fit_one_point`中。  
该函数接受一个`SinglePoint`参数，并使用函数
<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;a&space;\log(|bx&space;&plus;&space;ep|)&space;&plus;&space;c\log(|dx&plus;ep|)^2&space;&plus;&space;e\log(|fx&plus;ep|)^3&space;&plus;&space;g\log(|hx|&plus;ep)^4&space;&plus;&space;i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;a&space;\log(|bx&space;&plus;&space;ep|)&space;&plus;&space;c\log(|dx&plus;ep|)^2&space;&plus;&space;e\log(|fx&plus;ep|)^3&space;&plus;&space;g\log(|hx|&plus;ep)^4&space;&plus;&space;i" title="f(x) = a \log(|bx + ep|) + c\log(|dx+ep|)^2 + e\log(|fx+ep|)^3 + g\log(|hx|+ep)^4 + i" /></a>
进行拟合。  
在圆滑后的数据中，可以得到较好的拟合结果。
```python
from common.unit import SingleFile
from solutions.polyfit import fit_one_point

before_filepath = "data/origin/before/LINE_100_dbdt.dat"
after_filepath = "data/origin/after/new_LINE_100_dbdt.dat"

before_file = SingleFile(before_filepath)
after_file = SingleFile(after_filepath)

fit_one_point(before_file.get_one_point(), show=True)
```
  
