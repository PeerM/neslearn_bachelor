import sklearn
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

# dt = np.dtype(list([(i, np.uint8) for i in range(2048)]))
warpless = np.fromfile("../data/happylee4-smb-warpless.fm2_ram.bin", dtype=np.uint8, count=-1).reshape((-1, 2048))
print(warpless.shape)
warped = np.fromfile("../data/happylee-supermariobros,warped.fm2_ram.bin", dtype=np.uint8, count=-1).reshape((-1, 2048))
print(warped.shape)
glitchless = np.fromfile("../data/glitchless_mario_betr_then_adleikat.fm2_ram.bin", dtype=np.uint8, count=-1).reshape(
    (-1, 2048))
print(glitchless.shape)
glitchless_index = np.arange(start=0, stop=glitchless.shape[0])
warped_index = np.arange(start=0, stop=warped.shape[0])
warpless_index = np.arange(start=0, stop=warpless.shape[0])

used = warpless
uses_index = warpless_index
test = warped
test_index = warped_index

# clf = linear_model.LinearRegression()
# clf = linear_model.Lasso(max_iter=2000)
clf = linear_model.ElasticNet()
# clf = linear_model.SGDRegressor()
clf.fit(used, uses_index)

print(clf)

# Plot outputs
nth_sample = 1
plt.plot(test_index[0::nth_sample], clf.predict(test[0::nth_sample]), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
