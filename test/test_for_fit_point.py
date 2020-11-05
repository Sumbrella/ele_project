from common.unit import SingleFile
from common.functions import fit_point

before_filepath = "../data/origin/before/LINE_100_dbdt.dat"
after_filepath = "../data/origin/after/new_LINE_100_dbdt.dat"

before_file = SingleFile(before_filepath)
after_file = SingleFile(after_filepath)

fit_point(before_file.get_one_point(), show=True)