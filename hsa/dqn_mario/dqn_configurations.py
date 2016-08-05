from neon.initializers import Gaussian
from neon.layers import Affine
from neon.transforms import Rectlin

from hsa.simple_dqn.deepqnetwork import DeepQNetwork


class Deep3QNetwork(DeepQNetwork):
    def _createLayers(self, num_actions):
        # create network
        init_norm = Gaussian(loc=0.0, scale=0.01)
        layers = []
        layers.append(Affine(nout=1024, init=init_norm, activation=Rectlin(), batch_norm=self.batch_norm))
        layers.append(Affine(nout=2 ** 9 + 2 ** 8, init=init_norm, activation=Rectlin(), batch_norm=self.batch_norm))
        layers.append(Affine(nout=512, init=init_norm, activation=Rectlin(), batch_norm=self.batch_norm))
        # The output layer is a fully-connected linear layer with a single output for each valid action.
        layers.append(Affine(nout=num_actions, init=init_norm))
        return layers
