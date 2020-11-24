import sys
sys.path.append("/Users/sumbrella/Documents/GitHub/ele_project")

import pandas as pd

from ele_common.units import Generator


def generate(data_numbers, saved_path, file_name):  

    generator = Generator()

    for batch_id in range(data_numbers):
        
        print("[INFO] Generating data")

        time, res_data, origin_data, depth, res, square = generator.generate()
    
        print(time, res_data, origin_data, depth, res, square)

    return


if __name__ == '__main__':
    generate(1, 1, 1)
