def ram_to_numpy(ram_bytes):
    return np.frombuffer(ram_bytes, dtype=np.dtype(np.uint8)).reshape((32, 64))

def next_nram():
    for _ in range(60):
        emu.step()
    return ram_to_numpy(emu.get_ram())

fig = plt.figure()
plt.imshow(ram_to_numpy(emu.get_ram()), vmin=0, vmax=255, interpolation="nearest", cmap=plt.get_cmap('binary'))
ani = animation.FuncAnimation(fig, next_nram, interval=40, blit=True)

plt.ion()
for _ in range(4):
    for _ in range(60):
        emu.step()

    visualize_ram(emu.get_ram())
    plt.draw()
    # display.clear_output(wait=True)
    # display.display(plt.gcf())