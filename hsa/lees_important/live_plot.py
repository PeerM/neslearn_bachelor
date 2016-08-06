import os

# file_path = "./file.fifo"
# file_path = "../data/happylee4-smb-warpless.fm2_ram.bin"
from threading import Thread
import time
from matplotlib import pyplot as plt
import numpy as np

plt.ion()
ydata = [0] * 50
ax1 = plt.axes()
line, = plt.plot(ydata)

file_path = "D:/projekts/nes-ki/out.bin"


def update_plot(data):
    ydata.append(hash(data))
    del ydata[0]
    ymin = float(min(ydata)) - 10
    ymax = float(max(ydata)) + 10
    plt.ylim([ymin, ymax])
    line.set_xdata(np.arange(len(ydata)))
    line.set_ydata(ydata)  # update the data
    plt.draw()  # update the plot
    plt.pause(0.0001)


with open(file_path, "rb")as input_file:
    i = 0
    while True:
        input_file.readable()
        ram_set = input_file.read(1024)
        i += 1
        if len(ram_set) > 0:
            if i % 2 == 0:
                # print("more: ", len(ram_set))
                # print("tell: ", input_file.tell())
                update_plot(ram_set)
        else:
            print("empty")
            time.sleep(0.01)
