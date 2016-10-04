import socket

from pandas import DataFrame

from hsa import emu_connect

#config
prim_soc = socket.create_connection(("localhost", 9090))
out_filename="mario_1_1_third.hdf"

# open connection to emu
prim_soc.setblocking(True)
emu = emu_connect.Emu2(prim_soc)
emu.play_beginning()
emu.speed_mode("maximum")

# play movie and construct dataframes
ram_frames, inputs_list = emu.play_movie_with_output()
ram_frames_df = DataFrame.from_records(ram_frames)
inputs_df = DataFrame.from_dict(inputs_list)

# Save Dataframes to disk
inputs_df.to_hdf(out_filename,"inputs")
ram_frames_df.to_hdf(out_filename,"rams")