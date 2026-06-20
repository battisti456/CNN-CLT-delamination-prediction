import matplotlib.pyplot as plt
import pickle
from run.data_log import Data_Log
from svm import SVC_Data_Log
import numpy as np

LW = 2

OUT_TYPE = ('area','edge')

models_to_compare = [
    ('final10','final11','1DCNN',"#37BD95"),
    ('final1','final3','1DCNN+',"#B6BD37"),
    ('s:final1','s:final3','SVC',"#BD5F37"),
    ('final6','final5','2DCNN',"#3787BD"),
    ('final8','final9','Linear',"#BD3758"),
]


fig = plt.figure()
ax = fig.subplots()

for ma, me, title, color in models_to_compare:
    for i,m in enumerate((ma,me)):
        try:
            if m[:2] == 's:':
                dl_svm = SVC_Data_Log.load(m[2:])
                c0 = np.nanmean(dl_svm.metrics['test_class0_acc'])
                c1 = np.nanmean(dl_svm.metrics['test_class1_acc'])
            else:
                dl_ml = Data_Log.load(m)
                c0 = np.nanmean(dl_ml.metrics['test_class0_acc'][:,-1])
                c1 = np.nanmean(dl_ml.metrics['test_class1_acc'][:,-1])
            plt.scatter(
                c0,
                c1,
                c = color if i==0 else '#00000000',
                edgecolors=color,
                linewidths=LW,
                label = f'{title}' if m == ma else None
            )
        except FileNotFoundError  as e:
            print(e)

ax.set_xlabel('Pass Accuracy')
ax.set_ylabel('Fail Accuracy')
ax.legend(loc = 'lower left')
ax.axis('equal')

fig.set_size_inches(3.25,2.5)
fig.savefig('out/learn_report.svg',transparent=True)
fig.subplots_adjust(left = 0.2, bottom=0.2)
plt.show()