from hsa.v_keras.qlearning4k.games.game import Game
from .memory import ExperienceReplay
import numpy as np
import matplotlib.pyplot as plt
import os


def training_step(batch_size, learning_rate_gamma, memory, model):
    batch = memory.get_batch(model=model, batch_size=batch_size, gamma=learning_rate_gamma)
    # at the start of the training we might not have enough memories build up
    if batch:
        inputs, targets = batch
        batch_loss = float(model.train_on_batch(inputs, targets))
        return batch_loss
    return None


def make_epsilon_schedule(epsilon_start, epsilon_end, nb_epoch, epsilon_rate):
    delta = ((epsilon_start - epsilon_end) / (nb_epoch * epsilon_rate))
    final_epsilon = epsilon_end
    epsilon = epsilon_start
    for i in range(nb_epoch):
        yield epsilon
        if i / nb_epoch < epsilon_rate:
            epsilon = max(epsilon - delta, final_epsilon)


class Agent:
    def __init__(self, model, memory=None, memory_size=1000, nb_frames=None):
        assert len(model.output_shape) == 2, "Model's output shape should be (nb_samples, nb_actions)."
        if memory:
            self.memory = memory
        else:
            self.memory = ExperienceReplay(memory_size)
        if not nb_frames and not model.input_shape[1]:
            raise Exception("Missing argument : nb_frames not provided")
        elif not nb_frames:
            nb_frames = model.input_shape[1]
        elif model.input_shape[1] and nb_frames and model.input_shape[1] != nb_frames:
            raise Exception("Dimension mismatch : time dimension of model should be equal to nb_frames.")
        self.model = model
        self.nb_frames = nb_frames
        self.frames = None

    @property
    def memory_size(self):
        return self.memory.memory_size

    @memory_size.setter
    def memory_size(self, value):
        self.memory.memory_size = value

    def reset_memory(self):
        self.memory.reset_memory()

    def check_game_compatibility(self, game):
        game_output_shape = (1, None) + game.get_current_state().shape
        if len(game_output_shape) != len(self.model.input_shape):
            raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
        else:
            for i in range(len(self.model.input_shape)):
                if self.model.input_shape[i] and game_output_shape[i] and self.model.input_shape[i] != \
                        game_output_shape[i]:
                    raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
        if len(self.model.output_shape) != 2 or self.model.output_shape[1] != game.nb_actions:
            raise Exception('Output shape of model should be (nb_samples, nb_actions).')

    def get_game_data(self, frame):
        if self.frames is None:
            self.frames = [frame] * self.nb_frames
        else:
            self.frames.append(frame)
            self.frames.pop(0)
        return np.expand_dims(self.frames, 0)

    def clear_frames(self):
        self.frames = None

    def train(self, game: Game, nb_epoch=1000, batch_size=50, learning_rate_gamma=0.9, epsilon=(1., .1),
              epsilon_rate=0.5, play_period=1, action_repeat=1):
        # TODO IDEA move game into constructor or maybe not
        self.check_game_compatibility(game)
        # epsilon: can be single value or tuple for slope
        # TODO IDEA make epsilon handling more clear
        # current_epsilon = get_epsilon_schedule(epsilon)[current_epoch]
        if type(epsilon) in {tuple, list}:
            epsilon_schedule = list(make_epsilon_schedule(epsilon[0], epsilon[1], nb_epoch, epsilon_rate))
        else:
            epsilon_schedule = list(make_epsilon_schedule(epsilon, epsilon, nb_epoch, epsilon_rate))
        model = self.model
        # todo Unify this whole nb_actions thing
        nb_actions = model.output_shape[-1]
        win_count = 0
        for epoch in range(nb_epoch):
            loss = 0.
            game.reset()
            self.clear_frames()
            game_over = False
            S = self.get_game_data(game.get_current_state())
            cumulative_r = 0
            nr_training_sessions = 0
            while not game_over:
                for _1 in range(play_period):
                    # a: chosen action
                    # r: accomplished reward
                    # cumulative_r: performance metric to optimize
                    # q: the model returns the quality of each action as a number and we want the one it finds most prominent
                    cur_epsilon = epsilon_schedule[epoch]
                    if np.random.random() < cur_epsilon:
                        chosen_a = int(np.random.randint(game.nb_actions))
                    else:
                        q = model.predict(S)
                        chosen_a = int(np.argmax(q[0]))
                    play_result = game.play(chosen_a, action_repeat)
                    # actual_a is needed for of policy learning
                    actual_a = play_result.action
                    r = play_result.score
                    cumulative_r += r
                    game_over = play_result.is_over
                    S_prime = self.get_game_data(play_result.state)
                    self.memory.remember(S, actual_a, r, S_prime, game_over)
                    S = S_prime
                # TODO make this run in parallel
                batch_loss = training_step(batch_size, learning_rate_gamma, self.memory, model)
                if batch_loss:
                    loss += batch_loss
                    nr_training_sessions += 1
            # if game.is_won():
            #     win_count += 1
            # TODO IDEA better output logging function
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Reward {:.1f} | Epsilon {:.2f} | Avg Loss {:.2f}"
                  .format(epoch + 1, nb_epoch, loss, cumulative_r, epsilon_schedule[epoch], loss / nr_training_sessions))

    # TODO IDEA unify
    def play(self, game, nb_epoch=10, epsilon=0., visualize=True):
        self.check_game_compatibility(game)
        model = self.model
        win_count = 0
        frames = []
        for epoch in range(nb_epoch):
            game.reset()
            self.clear_frames()
            S = self.get_game_data(game)
            if visualize:
                frames.append(game.draw())
            game_over = False
            while not game_over:
                if np.random.rand() < epsilon:
                    print("random")
                    action = int(np.random.randint(0, game.nb_actions))
                else:
                    q = model.predict(S)
                    action = int(np.argmax(q[0]))
                game.play(action)
                S = self.get_game_data(game)
                if visualize:
                    frames.append(game.draw())
                game_over = game.is_over()
            if game.is_won():
                win_count += 1
        print("Accuracy {} %".format(100. * win_count / nb_epoch))
        if visualize:
            if 'images' not in os.listdir('.'):
                os.mkdir('images')
            for i in range(len(frames)):
                plt.imshow(frames[i], interpolation='none')
                plt.savefig("images/" + game.name + str(i) + ".png")
