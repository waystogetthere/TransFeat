import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

import glob
x = []
F1 = []
for fn in glob.glob('test_*'):
    del_num = fn.split('_')[1].split('.')[0]
    print(fn, del_num)
    with open(fn, 'rb') as f:
        log = pkl.load(f)
        print(log)
    s = log['F1_']
    s = [np.float(item) for item in s]

    x.append(int(del_num))
    F1.append(np.max(s))
x = np.array(x).reshape(-1,1)
F1 = np.array(F1).reshape(-1,1)
all_ = np.concatenate((x, F1), axis=-1)

all_ = all_[all_[:, 0].argsort()]
print(all_[:,0])
print(all_[:, 1])
plt.figure()
plt.plot(all_[:, 0], all_[:, 1]/all_[0, 1], marker='x')
plt.savefig('F1_score.png', dpi=800)