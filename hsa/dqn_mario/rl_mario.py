import random
import socket
import uuid

import hsa.simple_dqn.replay_memory as replay_memory
import hsa.reward_evaluation as re
import pandas
import numpy as np

from hsa.dqn_mario.dqn_argparse import parse_args
from hsa.dqn_mario.dqn_input import numpy_to_dqn, dqn_to_py
from hsa.emu_connect import Emu2
from hsa.simple_dqn.deepqnetwork import DeepQNetwork
from hsa.dqn_mario.dqn_configurations import Deep3QNetwork
from neon.backends import gen_backend

_random = random.random()


def process_raw_frame(last_frame):
    return np.frombuffer(last_frame, dtype=np.uint8).reshape((1, 2048))


class RlMarioPlayer(object):
    def __init__(self, weights_file, sim_speed=None):
        self.sim_speed = sim_speed
        args = parse_args("")
        args.batch_size = 1
        self.the_memories = replay_memory.ReplayMemory(60 * 60 * 20, args)
        gen_backend(backend="gpu", batch_size=1)
        self.dqn = Deep3QNetwork(255, args)
        self.dqn.load_weights(weights_file)
        prim_soc = socket.create_connection(("localhost", 9090))
        self.emu = Emu2(prim_soc)
        self.rewarder = re.MultiReward(re.MarioDeath(), re.MarioScore(), re.MarioXAcceleration())
        self.exploration_rate = 0.1
        self.train_for_x_minutes = 20
        self.single_play_period = 60

    def run(self):
        epoch = 0
        self.emu.load_slot(10)
        if self.sim_speed:
            self.emu.speed_mode(self.sim_speed)
        self.emu.step()
        last_frame = self.emu.get_ram()
        try:
            for i in range(int(60 / self.single_play_period * 60 * self.train_for_x_minutes)):
                # Play for a "Play"
                for j in range(self.single_play_period):
                    dqn_input_scores = self.dqn.predict(process_raw_frame(last_frame))
                    if random.random() > self.exploration_rate:
                        chosen_dqn_input = dqn_input_scores.argmax()
                    else:
                        chosen_dqn_input = random.randrange(0, 255)
                    py_input = dqn_to_py(chosen_dqn_input)
                    reward = self.rewarder.reward(last_frame)
                    current_frame = self.emu.full_step(py_input)
                    terminal = self.is_game_over(last_frame)
                    self.the_memories.add(chosen_dqn_input, reward, np.frombuffer(current_frame, dtype=np.uint8),
                                          terminal)
                    # End of episode reload savestate (maybe reset emulator later)
                    if terminal:
                        self.emu.load_slot(10)
                        epoch += 1

                    # move to next cycle
                    last_frame = current_frame
                # do some experience replay
                for j in range(int(self.single_play_period / 4)):
                    minibatch = self.the_memories.getMinibatch()
                    # train the network
                    self.dqn.train(minibatch, epoch)
        except NotADirectoryError:
            pass
        # except KeyboardInterrupt:
        #     self.emu.close()
        return

    def is_game_over(self, ram):
        return ram[0x075A] == 0


if __name__ == "__main__":
    # Nice weights files
    # "dqn_weights/1Layer/second_dqn_weights"
    # "dqn_weights/3Layer/d3_1_dqn_weights"
    ai = RlMarioPlayer("dqn_weights/rl_training/d3q_3rd_youtubevideo", "turbo")
    try:
        ai.run()
    finally:
        ai.dqn.save_weights("dqn_weights/rl_training/{}".format(uuid.uuid1()))
