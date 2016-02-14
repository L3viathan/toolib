from collections import namedtuple
import csv

class CSV(object):
    """Simple CSV wrapper for a file-like object."""
    def __init__(self, fileobject, **kwargs):
        """Initialize a CSV object.

        Optional keyword arguments:
            delimiter: The seperation character (default: ',')
            header: Whether the CSV file has a header (default: True)
        """
        self.fileobject = fileobject
        if self.fileobject.readable():
            arguments = {
                    "delimiter": kwargs.get("delimiter", ","),
                    "quotechar": kwargs.get("quotechar", '"'),
                    }
            self.reader = csv.reader(self.fileobject, **arguments)
            if kwargs.get("header", True):
                data = list(CSV.makeidentifier(next(self.reader)))
                self.Row = namedtuple("Row", data)
            else:
                self.Row = None # get later
        elif self.fileobject.writable():
            arguments = {
                    "delimiter": kwargs.get("delimiter", ","),
                    "quotechar": kwargs.get("quotechar", '"'),
                    }
            self.writer = csv.writer(self.fileobject, **arguments)
        else:
            raise OSError("File neither readable nor writable")

    def readline(self):
        """Return a namedtuple."""
        if self.Row is None:
            data = next(self.reader)
            self.Row = namedtuple("Row", ["col" + str(i) for i,_ in enumerate(data)])
            return self.Row(*data)
        else:
            return self.Row(*next(self.reader))

    def read(self):
        """Return a list of namedtuples."""
        lines = []
        for line in self:
            lines.append(line)
        return lines

    def __iter__(self):
        return self

    def __next__(self):
        return self.readline()

    def write(self, line):
        """Writes a single line to the file."""
        self.writer.writerow(line)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.fileobject.close()

    @staticmethod
    def makeidentifier(some_list):
        for item in some_list:
            try:
                int(item[0])
                yield "_" + item.replace(" ", "")
            except:
                yield item.replace(" ", "")

if __name__ == '__main__':
    with CSV(open("testfile.csv", "r")) as c:
        for line in c:
            print(line)
    with CSV(open("toastfile.csv", "w")) as c:
        c.write(["This", "is", "a", "test"])
        for element in range(5):
            c.write([element]*4)
