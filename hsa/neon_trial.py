from neon.util.argparser import NeonArgparser

parser = NeonArgparser(__doc__)
args = parser.parse_args()

from neon.data import load_mnist

(X_train, y_train), (X_test, y_test), nclass = load_mnist()

from neon.data import ArrayIterator

# setup training set iterator
train_set = ArrayIterator(X_train, y_train, nclass=nclass)
# setup test set iterator
test_set = ArrayIterator(X_test, y_test, nclass=nclass)

from neon.initializers import Gaussian

init_norm = Gaussian(loc=0.0, scale=0.01)


from neon.layers import Affine
from neon.transforms import Rectlin, Softmax

layers = []
layers.append(Affine(nout=100, init=init_norm, activation=Rectlin()))
layers.append(Affine(nout=10, init=init_norm,
                     activation=Softmax()))

# initialize model object
from neon.models import Model

mlp = Model(layers=layers)

from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti

cost = GeneralizedCost(costfunc=CrossEntropyMulti())


from neon.optimizers import GradientDescentMomentum

optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)


from neon.callbacks.callbacks import Callbacks

callbacks = Callbacks(mlp, eval_set=test_set, **args.callback_args)

mlp.fit(train_set, optimizer=optimizer, num_epochs=1, cost=cost,
        callbacks=callbacks)

results = mlp.get_outputs(test_set)
print(results)