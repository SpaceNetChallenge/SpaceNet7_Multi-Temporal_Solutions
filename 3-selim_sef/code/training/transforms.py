import random
import time

import albumentations as A


class RandomSizedCropAroundBbox(A.RandomSizedCrop):
    @property
    def targets_as_params(self):
        return ['rectangles', 'image']

    def get_params_dependent_on_targets(self, params):
        rectangles = params['rectangles']
        img_height, img_width = params['image'].shape[:2]
        rm = random.Random()
        rm.seed(time.time_ns())
        crop_height = rm.randint(self.min_max_height[0], self.min_max_height[1])
        crop_width = int(crop_height * self.w2h_ratio)

        if rectangles:
            x, y, w, h = rm.choice(rectangles)
            min_x_start = max(x + (w / 2 if w >= crop_width else w) - crop_width, 0)
            min_y_start = max(y + (h / 2 if h >= crop_height else h) - crop_height, 0)
            max_x_start = min(x + (w / 2 if w >= crop_width else 0), img_width - crop_width)
            max_y_start = min(y + (h / 2 if h >= crop_height else 0), img_height - crop_height)
            if max_x_start < min_x_start:
                min_x_start, max_x_start = max_x_start, min_x_start
            if max_y_start < min_y_start:
                min_y_start, max_y_start = max_y_start, min_y_start
            start_y = rm.randint(int(min_y_start), int(max_y_start)) / img_height
            start_x = rm.randint(int(min_x_start), int(max_x_start)) / img_width
        else:
            start_y = rm.random()
            start_x = rm.random()
        return {'h_start': (start_y * img_height) / (img_height - crop_height),
                'w_start': (start_x * img_width) / (img_width - crop_width),
                'crop_height': crop_height,
                'crop_width': int(crop_height * self.w2h_ratio)}