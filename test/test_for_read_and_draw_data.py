import matplotlib.pyplot as plt
from ele_common.units import SingleFile

# before_filepath = '../data/origin/before/LINE_100_dbdt.dat'
# after_filepath = '../data/origin/after/new_LINE_100_dbdt.dat'

before_filepath = "../data/generate/data/LINE_001_dbdt.dat"
after_filepath = "../data/generate/teacher/NEW_LINE_001_dbdt.dat"

before_file = SingleFile(before_filepath)
after_file = SingleFile(after_filepath)

# read the first point
before_point = before_file.get_one_point()
after_point = after_file.get_one_point()

before_point.plot(show=False)
after_point.plot(show=False)

print(
f"""
after_point_size: {after_point._size}
before_point_size: {before_point._size}
"""
)
print(after_point.get_data())

plt.show()
