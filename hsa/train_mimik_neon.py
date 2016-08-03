import pandas
from neon.layers import Affine, BatchNorm
from neon.transforms import Rectlin, Tanh
from neon.initializers import Gaussian
from neon.models import Model
from neon.layers import GeneralizedCost
from neon.transforms import SumSquared, SmoothL1Loss
from neon.optimizers import GradientDescentMomentum
from neon.data import ArrayIterator
from neon.backends import gen_backend
from neon.callbacks.callbacks import Callbacks
from hsa.Costs import SumSquaredWithL1Loss

inputs = pandas.read_hdf("warpless.hdf", key="inputs")[:10000]
rams = pandas.read_hdf("warpless.hdf", key="rams")[:10000]

# gen_backend(backend='cpu', batch_size=128)
gen_backend(backend='cpu', batch_size=2048)

train = ArrayIterator(rams.values, inputs.values, make_onehot=False)

init_norm = Gaussian(loc=0.0, scale=0.01)
layers = []
layers.append(Affine(nout=1024, init=init_norm, activation=Rectlin(), name="first"))
layers.append(BatchNorm(rho=0.9, eps=1e-3))
# layers.append(Affine(nout=256, init=init_norm, activation=Rectlin()))
# layers.append(BatchNorm(rho=0.9, eps=1e-3))
# layers.append(Affine(nout=64, init=init_norm, activation=Rectlin()))
layers.append(BatchNorm(rho=0.9, eps=1e-3))
layers.append(Affine(nout=8, init=init_norm, activation=Tanh()))

mlp = Model(layers=layers)

optimizer = GradientDescentMomentum(0.001, momentum_coef=0.9)
cost = GeneralizedCost(costfunc=SumSquared())
callbacks = Callbacks(mlp)

mlp.fit(train, optimizer=optimizer, num_epochs=200, cost=cost,
        callbacks=callbacks)
mlp.save_params("warpless1.prm")
