import sys


class TimeProgressBar:
    def __init__(self, max, time, w=40, label='Progress'):
        self.max = max
        self.w = w
        self.label = label
        self.time = time

    def print(self, v):
        sys.stdout.write(self.label + " [")
        c = round(v * self.w / self.max)
        d = self.w - c
        sys.stdout.write("#" * c)
        sys.stdout.write(" " * d)
        sys.stdout.write("] %.02f%%\n" % (v * 100.0 / self.max))

