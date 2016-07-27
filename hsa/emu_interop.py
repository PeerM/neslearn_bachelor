import os

# file_path = "./file.fifo"
# file_path = "../data/happylee4-smb-warpless.fm2_ram.bin"
from threading import Thread
import time
from matplotlib import pyplot as plt

file_path = "D:/projekts/nes-ki/out.bin"
with open(file_path, "rb")as input_file:
    i = 0
    while True:
        input_file.readable()
        ram_set = input_file.read(1024)
        i += 1
        if len(ram_set) > 0:
            # if i % 10 == 0:
            print("more: ", len(ram_set))
        else:
            print("empty")
            time.sleep(0.01)
