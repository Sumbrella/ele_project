import os.path

from common.unit import SinglePoint


class SingleFile:

    def __init__(self, filepath, pre_read_number=None):

        if not os.path.exists(filepath):
            raise ValueError('Don\'t find the file {}'.format(filepath))

        if pre_read_number is None:
            pre_read_number = 0

        # private
        self._point_number = None
        self._remind_point_number = None
        self._pre_reader_number = pre_read_number
        self._fp = open(filepath, 'r')
        self._point_reader = None
        # public
        path, self.filename = os.path.split(filepath)
        self.filename = self.filename.split('.', 1)[0]
        # init
        self._init_title(self._fp)
        self._init_point_reader()

    @property
    def point_reader(self):
        return self._point_reader

    @property
    def date(self):
        return self._date

    @property
    def point_number(self):
        return self._point_number

    @property
    def remind_point_number(self):
        return self._remind_point_number

    def _init_title(self, fp):

        self._point_number = eval(fp.readline())
        self._remind_point_number = self._point_number
        self._date = fp.readline()

    def _init_point_reader(self):
        def reader():
            while True:
                try:
                    yield SinglePoint(fp=self._fp, skip_line=2)
                except Exception as e:
                    # print(e)
                    # raise
                    break
            yield None

        self._point_reader = reader

    def _read_point(self):

        if self._point_number is None:
            raise ValueError('in SingleFile._read_point, no point_number existed, please read point_number first')
        point = SinglePoint(fp=self._fp, skip_line=2)

        return point

    def get_reader(self, batch_size=10):

        def reader():
            points = []
            for point_id, point in enumerate(self.point_reader()):

                if point is None:
                    break

                points.append(point)

                if len(points) is batch_size:
                    yield points
                    points = []

            if len(points):
                yield points

        return reader

    def get_one_point(self) -> SinglePoint:
        point = next(self.point_reader())
        if point is None:
            raise ValueError('No more point in this file')
        return point

    def __describe(self):
        # TODO：ADD FILE BASIC DESCRIPTION
         (
            f"""
============================{self.filename}============================
point_number: {self._point_number}
============================{'='*len(self.filename)}============================"

            """
        )


if __name__ == '__main__':
    from common.unit import SingleFile

    singlefile = SingleFile(filepath='../../data/origin/before/LINE_120_dbdt.dat')
    print(singlefile._date)
    print(singlefile.filename)
    point = singlefile.get_one_point()
    point.plot()
