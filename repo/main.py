#File was edited to run different analyses. Not ready for any sort of direct use.


from torch.nn import functional as F
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from module.module1 import Module1
from module.module2 import Module2
from module.module3 import Module3
from module.module4 import Module4, Module5, Module6, Module7
from module.module8 import Module8
from module.module9 import Module9
from module.module10 import Module10
from module.module12 import Module12
from module.module13 import Module13, Module14, Module15, Module16, Module17
from module.module18 import Module18, Module19
from module.module20 import Module20
from module.module21 import Module21
from module.module22 import Module22
from module.module23 import Module23
from module.module24 import Module24
from module.module25 import Module25
from run.hyperparameters import Hyperparameters
#from run.run1 import run1
from run.run2 import run2
from run.run3 import run3
from common.loss import binary_focal_loss

NUM_EPOCHS = 100

hp = Hyperparameters(
    module = Module25,
    name = 'gbm6',
    ignore_proper_folds=True,
    out_type = 'edge',
    num_folds = 3,
    append_poster_to_train = True,
    batch_size = 64,
    butter_order = 6,
    classification_thresholds = (0.05,),
    classify = True,
    cutoff_frequency = 300000,
    data_balance_mode = 'match',
    dropout = 0.05,
    early_exit_loss = None,
    freeze_encoder_epochs = 0,
    in_interpolation_mode = 'none',
    loss_config = {'alpha': 1, 'gamma': 2},
    loss_function = binary_focal_loss,
    lr_scheduler = CosineAnnealingWarmRestarts,
    lr_scheduler_config=dict(
        T_0 = NUM_EPOCHS,
        eta_min = 0.001,
    ),
    manual_class_weight = None,
    nan_policy = 'remove',
    num_epochs = NUM_EPOCHS,
    optimizer = SGD,
    optimizer_config = {'momentum': 0.9, 'weight_decay': 0.0001, 'lr': 0.1},
    out_boost_amp = 1,
    out_shape = 'block',
    ply = 'both',
    reorder_layups = (6, 13, 3, 17, 9, 15, 1, 22, 4, 16, 5, 20, 10, 23, 11, 12, 2, 19, 7, 14, 8, 18, 0, 21),#type:ignore
    shuffle = True,
    skip_incomplete_folds = True,
    use_parameters=('signal','fft','num_layers')
)
if __name__ == '__main__':
    NUM_EPOCHS = 100
    run2(hp)
    # run3(
    #     Hyperparameters(
    #         module = Module23,
    #         loss_function=binary_focal_loss,
    #         loss_config = dict(
    #             alpha = 1,
    #             gamma = 2
    #         ),
    #         optimizer_config=dict(
    #             momentum=0.9,
    #             weight_decay=1e-4,
    #             lr = 0.05
    #         ),
    #         lr_scheduler=CosineAnnealingWarmRestarts,
    #         lr_scheduler_config=dict(
    #             T_0 = NUM_EPOCHS,
    #             eta_min = 0.001,
    #         ),
    #         #manual_class_weight = (0.5,1),
    #         name = 'final6', 
    #         num_folds=6,
    #         ply = 'both',
    #         batch_size=32,
    #         num_epochs=NUM_EPOCHS,
    #         out_type='area',
    #         data_balance_mode='match',
    #         out_boost_amp=1,
    #         dropout=0.2,
    #         skip_incomplete_folds=True,
    #         reorder_layups='auto',
    #         transfer_learning = 'veil5',
    #         transfer_learning_transform = 'spec_rect_mask',
    #         transfer_learning_boost = 1,
    #         transfer_learning_transform_config = dict(
    #             portion_to_cover = 0.7
    #         ),
    #         transfer_epochs = 300,
    #         use_parameters=('spectrogram',),
    #         freeze_encoder_epochs=10,
    #         append_poster_to_train=True
    # ))

# if __name__ == '__main__':
#     run2(hp)