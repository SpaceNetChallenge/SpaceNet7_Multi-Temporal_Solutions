# Instructions for Model Development

This section provides instructions for the model development phase.

## Download SpaceNet-7 data

```
# prepare data directory
DATA_DIR=${HOME}/data/spacenet7/spacenet7
mkdir -p ${DATA_DIR}
cd ${DATA_DIR}

# download and extract train data
aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz .
tar -xvf SN7_buildings_train.tar.gz

# download and extract test data
aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz .
tar -xvf SN7_buildings_test_public.tar.gz
```

## Prepare training environment

```
# git clone source
PROJ_DIR=${HOME}/spacenet7_solution
git clone git@github.com:motokimura/spacenet7_solution.git ${PROJ_DIR}

# build docker image
cd ${PROJ_DIR}
./docker/build.sh

# launch docker container
ENV=desktop1
./docker/run.sh ${ENV}
```

## Preprocess dataset

All commands below have to be executed inside the container.

```
./tools/geojson_to_mask.py

./tools/split_dataset.py
```

Optional:

```
./tools/split_dataset_random.py

./tools/filter_small_polygons.py
```

## Train segmentation models

All commands below have to be executed inside the container.

```
EXP_ID=9999  # new experiment id
./tools/train_spacenet7_model.py [--config CONFIG_FILE] EXP_ID ${EXP_ID}
```

## Test segmentation models

All commands below have to be executed inside the container.

```
EXP_ID=9999  # previous experiment id from which config and weight are loaded
./tools/test_spacenet7_model.py [--config CONFIG_FILE] --exp_id ${EXP_ID}
```

## Ensemble segmentation models

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999,9998,9997,9996,9995]'  # previous experiments used for ensemble
./tools/ensemble_models.py [--config CONFIG_FILE] ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```

## Refine predicted masks

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999,9998,9997,9996,9995]'  # previous experiments used for ensemble
./tools/refine_pred_mask.py [--config CONFIG_FILE] ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```

## Convert predicted mask to polygons

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999,9998,9997,9996,9995]'  # previous experiments used for ensemble
./tools/pred_mask_to_poly.py [--config CONFIG_FILE] ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```

## Track polygons

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999,9998,9997,9996,9995]'  # previous experiments used for ensemble
./tools/track_polys.py [--config CONFIG_FILE] ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```

## Test segmentation models (val)

All commands below have to be executed inside the container.

```
EXP_ID=9999  # previous experiment id from which config and weight are loaded
./tools/test_spacenet7_model.py --config configs/test_to_val_images.yml --exp_id ${EXP_ID}
```

## Ensemble segmentation models (val)

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999,]'  # experiment used for the testing
./tools/ensemble_models.py --config configs/test_to_val_images.yml ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```

## Refine predicted masks (val)

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999,]'  # experiment used for the testing
./tools/refine_pred_mask.py --config configs/test_to_val_images.yml ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```

## Convert prerdicted mask to polygons (val)

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999,]'  # experiment used for the testing
./tools/pred_mask_to_poly.py --config configs/test_to_val_images.yml ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```

## Track polygons (val)

All commands below have to be executed inside the container.

```
ENSEMBLE_EXP_IDS='[9999,]'  # experiment used for the testing
./tools/track_polys.py --config configs/test_to_val_images.yml ENSEMBLE_EXP_IDS ${ENSEMBLE_EXP_IDS}
```
