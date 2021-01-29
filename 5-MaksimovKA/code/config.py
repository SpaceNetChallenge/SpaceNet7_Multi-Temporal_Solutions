import albumentations as albu
import segmentation_models_pytorch as smp

logs_path = '/wdata/segmentation_logs/'
folds_file = '/wdata/folds.csv'
load_from = '/wdata/segmentation_logs/fold_1_siamse-senet154/checkpoints/best.pth'
multiplier = 5

main_metric = 'dice'
minimize_metric = False
device = 'cuda'
val_fold = 1
folds_to_use = (2, 3, 4, 5, 6, 7, 8)
n_classes = 3
input_channels = 3
crop_size = (320, 320)
val_size = (1024, 1024)

batch_size = 4
num_workers = 4
val_batch_size = 1

shuffle = True
lr = 1e-4
momentum = 0.0
decay = 0.0
loss = 'focal_dice'
optimizer = 'adam_gcc'
fp16 = False

alias = 'fold_'
model_name = 'siamse-senet154'
scheduler = 'steps'
steps = [15, 25]
step_gamma = 0.25
augs_p = 0.5
best_models_count = 5
epochs = 30
weights = 'imagenet'
limit_files = None

preprocessing_fn = smp.encoders.get_preprocessing_fn('senet154', weights)

train_augs = albu.Compose([albu.OneOf([albu.RandomCrop(crop_size[0], crop_size[1], p=1.0)
                                       ], p=1.0),
                           albu.Flip(p=augs_p),
                           albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=augs_p),
                           ], p=augs_p)

valid_augs = albu.Compose([albu.PadIfNeeded(min_height=val_size[0], min_width=val_size[1], p=1.0)])