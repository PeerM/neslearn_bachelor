import collections
from io import BytesIO

import imageio
import matplotlib
import numpy as np
from PIL import ImageFont, Image, ImageDraw
from scipy import misc

import hsa
import hsa.ba.rewards
import hsa.machine_constants
from extern.fceux_learningenv.nes_python_interface.nes_python_interface import NESInterface, RewardTypes
from hsa.nes_python_input import *
from hsa.visualization.parse_fm2 import parse_fm2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# unused
def plot_rewards(rewards):
    # Make a random plot...
    fig = plt.figure()

    plt.plot(rewards)
    fig.tight_layout(pad=0)

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    # fig.canvas.draw()

    imgdata = BytesIO()
    plt.savefig(imgdata, format='png')
    plt.clear()
    imgdata.seek(0)  # rewind the data
    # im = misc.imread(imgdata)
    im = imageio.imread(imgdata)
    imgdata.close()
    return im


# nes.reset_game() # only needed if the movie skips the beginning

def play_and_record(movie_path,video_name):
    movie_writer = imageio.get_writer(video_name, fps=60, quality=9)
    nes = NESInterface(hsa.machine_constants.mario_rom_location, eb_compatible=False,
                       auto_render_period=3)
    with open(movie_path) as movie_file:
        inputs_from_movie = list(parse_fm2(movie_file))

    rewards = collections.deque(maxlen=24)  # one minute
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)

    for i, combi in enumerate(inputs_from_movie):
        reward = nes.act(py_to_nes_wrapper(combi))
        if i < 6:
            reward = 0  # ignore the first few frames on the start menu, because the rewards are not valid there
        rewards.append(reward)
        # video_frame = stack_images((208,208),[nes.getScreenRGB(),plot_rewards(rewards)])
        video_frame = nes.getScreenRGB()
        video_frame = misc.imresize(video_frame, 2.0, "nearest")
        img = Image.fromarray(video_frame)
        width, height = img.size
        bigger_img = Image.new("RGB",(width+32,height))
        bigger_img.paste(img,(0,0))
        img = bigger_img
        draw = ImageDraw.Draw(img)
        for i, reward in enumerate(rewards):
            draw.text((width, i * 18), "{}".format(reward), (255, (i + 1) * 14, 0), font=font)
        draw = ImageDraw.Draw(img)
        video_frame = np.array(img)
        movie_writer.append_data(video_frame)
    del nes
    movie_writer.close()


def main():
    # play_and_record(movie_path="../movies/5_1-1_without-shortcut.fm2", video_name="default.mp4",reward_func=hsa.ba.rewards.make_main_reward())
    play_and_record(movie_path="../movies/5_1-1_without-shortcut.fm2", video_name = "5_ehrenbrav.mp4")


if __name__ == '__main__':
    main()
