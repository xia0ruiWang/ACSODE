import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
import matplotlib as mpl
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
mpl.rcParams['text.usetex'] = True


# Make data.

data_test = pd.read_csv(r"D:\pycharmproject\Augment node for PMSM\SPMSMzhengxiepo500hou.csv", header=None)
x = np.array(data_test.values)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_real = torch.from_numpy(x)

obsc = x_real.cpu()
fig = plt.figure()
mpl.rcParams['axes.unicode_minus'] = True
font = {'family': 'Times New Roman',
        'weight': 800,
        'size': 18,
        }
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r'$i_\alpha$[$A$]', fontdict=font)
ax.set_ylabel(r'$i_\beta$[$A$]', fontdict=font)
z = np.array([o.detach().numpy() for o in obsc])
ax.plot(z[:, 0], z[:, 1], color='b', alpha=0.5)
'''plt.legend(loc='best')'''
# time.sleep(0.1)
'''plt.savefig(r"IPMSMzhengjieyue.png", format="png")'''
plt.show()