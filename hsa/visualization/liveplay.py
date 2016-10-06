import socket

from hsa import emu_connect
from hsa.reward_evaluation import mario_x_speed
from hsa.visualization.liveplot import LivePlotter

prim_soc = socket.create_connection(("localhost", 9090))
prim_soc.setblocking(True)
emu = emu_connect.Emu2(prim_soc)

#plotter = LivePlotter()

def visualize_ram(ram):
    print(mario_x_speed(ram))

try:
    for _ in range(400):
        for _ in range(10):
            emu.step()
        visualize_ram(emu.get_ram())
except KeyboardInterrupt:
    pass
    #emu.close()