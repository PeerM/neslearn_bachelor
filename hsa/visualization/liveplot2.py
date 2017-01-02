import matplotlib
matplotlib.use("GTK3Agg")
import threading
import random
from hsa.visualization.liveplot import LivePlotter

class Plotter(object):
    def __init__(self, max_y = None, interval=250, history_length = 50 ):
        super().__init__()
        self.max_y = max_y
        self.plotter = LivePlotter(interval=interval, size= history_length)
        self.thread = threading.Thread(daemon=True, target=self.plotter.start,name="LivePlotter")
        self.i = 0

    def start(self):
        self.thread.start()

    def stop(self):
        self.plotter.stop()

    def plot(self,value):
        # self.plotter.xs.append(self.i)
        # self.i += 1
        self.plotter.ys.append(value)

def main():
    import time
    plotter = Plotter()
    plotter.start()
    for i in range(8000):
        plotter.plot(random.random())
        time.sleep(0.5)
    plotter.stop()

if __name__ == "__main__":
    main()

