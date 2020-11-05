from paddle import fluid
import numpy as np

with fluid.dygraph.guard():
    conv1 = fluid.dygraph.Conv2D(num_channels=1, num_filters=5, filter_size=3, stride=1, padding=(0,1))
    x = np.random.randn(1, 1, 3, 3).astype('float32')
    x = fluid.dygraph.to_variable(x)
    res = conv1(x)
    print(res)
