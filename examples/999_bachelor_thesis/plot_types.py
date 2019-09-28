# This script requires the python package seaborn
# please run pip install seaborn before running this script

import numpy as np
import seaborn
import matplotlib.pyplot as plt

np.random.seed(123)
data = np.random.normal(loc=0, scale=1, size=25)

seaborn.distplot(data, hist=True, kde=True, rug=True)

plt.show()
