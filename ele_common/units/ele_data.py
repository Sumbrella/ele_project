#!/usr/bin/env python
# Author: Sumbrella
# Created Time: Mon Nov 23 22:35:34 2020

import csv


class EleData:

    def __init__(self, ele_data):
        self.params = [
            'origin_data',
            'result_data',
            'time',
            'layer_number',
            'depths',
            'res',
        ]
        
        self.origin_data = ele_data['origin_data']
        self.result_data = ele_data['result_data']
        self.time = ele_data[':wqtime']
        self.layer_number = ele_data['layer_number']
        self.depths = ele_data['depths']
        self.res = ele_data['res']
        
    
    def to_csv(csv_file):
        with open(csv_file, 'a+', newline='') as fp:
            writer = csv.writer(df)
            writer.writerow(
                self.origin_data,
                self.result_data,
                self.time,
                self.layer_numebr,
                self.depths,
                self.res
            )



if __name__ == '__main__':
    from ele_common.units import Generator
    generator = Generator()

    data = generator.generate()

