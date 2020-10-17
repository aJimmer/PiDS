import time

class FileTailer(object):
    def __init__(self, filepath, sleep=50, full_read=True, tail=True):
        self._filepath = filepath
        self._sleep = sleep * 1e-3
        self._full_read = full_read
        self._tail = tail

    def readlines(self, offset=0):
        try:
            with open(self._filepath) as fp:

                if self._full_read:
                    fp.seek(offset)
                    for row in fp:
                        yield row

                # Okay now dynamically tail the file
                if self._tail:
                    while True:
                        current = fp.tell()
                        row = fp.readline()
                        if row:
                            yield row
                        else:
                            fp.seek(current)
                            time.sleep(self._sleep)

        except IOError as err:
            print('Error reading the file {0}: {1}'.format(self._filepath, err))
            return
