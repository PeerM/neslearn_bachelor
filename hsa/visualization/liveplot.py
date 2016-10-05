import collections
from queue import Queue
from random import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')


class LivePlotter(object):
    def __init__(self, size=50, interval=1000 / 60):
        super().__init__()
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.xs = collections.deque([None] * size, maxlen=size)
        self.ys = collections.deque([None] * size, maxlen=size)
        self.interval = int(interval)

    def animate(self, i):
        self.xs.append(i)
        self.ys.append(random())
        self.ax1.clear()
        self.ax1.plot(self.xs, self.ys)

    def start(self):
        ani = animation.FuncAnimation(self.fig, self.animate, interval=self.interval)
        plt.show()


LivePlotter(100, 7).start()
