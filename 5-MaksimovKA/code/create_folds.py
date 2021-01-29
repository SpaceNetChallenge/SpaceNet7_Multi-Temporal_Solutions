import os
import fire
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def create_folds(images_path='/data/SN7_buildings/train/',
                 seed=769,
                 n_folds=8,
                 out_file='/wdata/folds.csv',
                 masks_path='/wdata/train_masks/'):
    aois = [el for el in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, el))]
    all_ids = np.array(sorted(aois))
    sub_targets = []
    kf = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
    for i, (_, evaluate_index) in enumerate(kf.split(all_ids)):
        # print(i)
        ids = all_ids[evaluate_index]
        print(i, len(ids), ids)
        for _id in ids:
            id_path = os.path.join(images_path, _id, 'images_masked')
            files = os.listdir(id_path)
            masks = [os.path.join(masks_path, _id, el) for el in files]
            files = [os.path.join(id_path, el) for el in files]

            target_df = {'image_path': [], 'mask_path': [], 'fold_number': []}
            target_df['image_path'] += files
            target_df['fold_number'] += [i + 1 for el in files]
            target_df['mask_path'] += masks

            target_df = pd.DataFrame(target_df)
            target_df = target_df[['image_path', 'mask_path', 'fold_number']]

            sub_targets.append(target_df)
    target_df = pd.concat(sub_targets)
    target_df.to_csv(out_file, index=False)
    print(target_df.head())


if __name__ == '__main__':
    fire.Fire(create_folds)
