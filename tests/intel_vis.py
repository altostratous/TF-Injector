import pickle
from matplotlib import pyplot as plt
import numpy as np

data = pickle.load(open('conv_v2_59.pkl', mode='rb'))

x = [1, 10, 100]

base_accuracy = data['nofault'][1][0]
for m in data:
    if m == 'nofault':
        continue
    y = []
    e = []
    for b in data[m]:
        b_ = data[m][b]
        b_ = [(base_accuracy - i) / base_accuracy for i in b_ if i != 1]
        sdc = sum(b_) / len(b_)
        std = np.std(b_)
        print(m, b, sdc)
        y.append(sdc)
        e.append(std)
    plt.errorbar(x, y, e, label=m, elinewidth=0.5, capsize=5)
plt.legend()
plt.title('SDC rate')
plt.xlabel('bit-flips')
plt.ylabel('sdc rate')
plt.show()