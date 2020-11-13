import pandas as pd
from solutions.polyfit.get_reader import get_reader

reader = get_reader(data_dir='../data/train/before', csv_dir='../data/train/teacher', batch_size=1)

for i, data in enumerate(reader()):
    data, label = data
    print(data.shape)
    print(label)
    break