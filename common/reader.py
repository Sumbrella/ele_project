from common.unit import SinglePoint

class Reader:

    def __init__(self, filename):
        self._filename = filename
        self._point_number = None
        self._remind_point_number = None
        self._date = None

        fp = open(filename, 'r')

        self._read_title(fp)

    def _read_title(self, fp):
        self._point_number = fp.readline()
        self._remind_point_number = self._point_number
        self._date = fp.readline()

    def get_reader(self, point_number=1):
        fp = open(self._filename, 'r')
        def reader():
            points = []

            if len(points) == point_number:
                pass

        return reader
