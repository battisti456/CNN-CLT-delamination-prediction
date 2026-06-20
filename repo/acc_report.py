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
    ('svc:final1','svc:final3','SVC',"#BD5F37"),
    ('final6','final5','2DCNN',"#3787BD"),
    ('final8','final9','Linear',"#BD3758"),
    ('gbm:gbm1','gbm:gbm2','GBM default',"#E800E4"),
    ('gbm:gbm3','gbm:gbm4','GBM Vahid folds',"#008D09"),
    ('gbm:gbm5','gbm:gbm6','GBM Vahid folds and params',"#FFCD62"),
]

fig = plt.figure()
ax = fig.subplots()

for ma, me, title, color in models_to_compare:
    for i,m in enumerate((ma,me)):
        try:
            if ':' in m:
                model,name  = m.split(':')
                dl_svm = SVC_Data_Log.load(name,model)
                f1 = np.mean(dl_svm.metrics['test_f1'])
                auroc = np.mean(dl_svm.metrics['test_auroc'])
            else:
                dl_ml = Data_Log.load(m)
                f1 = np.mean(dl_ml.metrics['test_f1'][:,-1])
                auroc = np.mean(dl_ml.metrics['test_auroc'][:,-1])
            plt.scatter(
                f1,
                auroc,
                c = color if i==0 else "#00000000",
                edgecolors=color,
                linewidths=LW,
                label = f'{title}' if m == ma else None
            )
        except FileNotFoundError:
            ...

ax.set_xlabel('F1 score')
ax.set_ylabel('AUROC score')
ax.legend()
ax.axis('equal')
fig.set_size_inches(3.25,2.5)
fig.savefig('out/acc_report.svg',transparent=True)
fig.subplots_adjust(left = 0.2, bottom=0.2)
plt.show()