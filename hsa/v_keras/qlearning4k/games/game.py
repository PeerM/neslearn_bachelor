# noinspection PyMethodMayBeStatic
class PlayResult(object):
    def __init__(self, state, score, action, is_over: bool, is_won: bool):
        super().__init__()
        self.action = action
        self.is_won = is_won
        self.is_over = is_over
        self.score = score
        self.state = state


class PlayResultWithImage(PlayResult):
    def __init__(self, state, score, is_over: bool, is_won: bool, image):
        super().__init__(state, score, is_over, is_won)
        self.image = image


class Game:
    def __init__(self, game_name, nb_actions):
        self.nb_actions = nb_actions
        self.game_name = game_name
        self.reset()

    def name(self):
        return self.game_name

    def nb_actions(self):
        return self.nb_actions

    def reset(self):
        pass

    def get_current_state(self):
        return None

    def play(self, action, nr_repeat_actions) -> PlayResult:
        pass
