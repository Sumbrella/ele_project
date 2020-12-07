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
        self.time = ele_data['time']
        self.layer_number = ele_data['layer_number']
        self.depths = ele_data['depths']
        self.res = ele_data['res']

    def to_csv(self, csv_file):
        with open(csv_file, 'a+', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow([
                self.origin_data,
                self.result_data,
                self.time,
                self.layer_number,
                self.depths,
                self.res
            ])


if __name__ == '__main__':
    import sys
    sys.path.append("/Users/sumbrella/Documents/GitHub/ele_project")
    import matplotlib.pyplot as plt
    from ele_common.units import Generator

    generator = Generator()
    data = generator.generate(debug=True)
    ele_data = EleData(data)

    print(ele_data.layer_number)
    print(ele_data.origin_data)
    print(ele_data.result_data)

    plt.plot(ele_data.time, ele_data.origin_data, label='origin')
    plt.plot(ele_data.time, ele_data.result_data, label='added')
    plt.legend()
    plt.show()

