import albumentations as A
import cv2
from albumentations import RandomCrop, CenterCrop

from training.transforms import RandomSizedCropAroundBbox


class InstanceAugmentations:
    def __init__(self):
        super().__init__()

    def create_train_transforms(self, crop_size):
        transforms = [
            A.OneOf(
                [
                    RandomCrop(crop_size, crop_size, p=0.3),
                    RandomSizedCropAroundBbox(min_max_height=(int(crop_size * 0.65), int(crop_size * 1.4)),
                                              height=crop_size,
                                              width=crop_size, p=0.7)
                ], p=1),
            A.Rotate(20, p=0.2, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.FancyPCA(p=0.2)
        ]
        return A.Compose(transforms, additional_targets={'labels': 'mask'})

    def create_val_transforms(self):
        transforms = [
            CenterCrop(2048, 2048)
        ]
        return A.Compose(transforms)
