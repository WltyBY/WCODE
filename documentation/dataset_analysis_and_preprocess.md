# Dataset Analysis

`DatasetFingerprintExtractor` in [wcode/preprocessing/dataset_analysis.py](../wcode/preprocessing/dataset_analysis.py) gathers dataset-wide statistical fingerprints to guide downstream preprocessing and inform network design at training time following [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/master).

This class performs:

1. Dataset splitting (train / val / (test) or k-fold).  
2. Statistics on the training and validation sets:  
   - target spacing for resampling;
   - per-channel intensity statistics for normalization;
   - median image size after zero-cropping.

These results are persisted to `dataset_split.json`, `dataset_fingerprint.json`, and `plans.json`. The implementation works for both medical and natural images (**natural-image path not yet validated**).

## Foreground Statistic

### 3D image

`DatasetFingerprintExtractor.collect_foreground_intensities` only processes 3D images at the channel level. 

**Note**

- When computing foreground-class pixel statistics, we ignore pixels belonging to [**background**, **ignore**, and **unlabel**, or just contain these characters]. For example, if you run a scribble labeled dataset and there is a class that will not compute loss on it, use the [**ignored** or **unlabeled**] to skip the class.

This means that if you do **not** want a certain class to be treated as foreground, you must edit the `labels` section in `dataset.yaml` and include the keywords **background**, **ignore**, or **unlabel** in the corresponding class name; any voxel whose label matches one of these keywords will be excluded from intensity statistics.

These keywords are **case-insensitive**; simply include them anywhere in the `labels`.

### 2D image

`DatasetFingerprintExtractor.statistic_for_2d_data` only processes 2D images at the channel level. For 2D images, we compute statistics over **all pixels** in every channel; the label is **not** used.

## Target Spacing (Only for 3D Images)

Following [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/master), we first compute the median spacing along each axis across all images.  
If the worst-resolution axis (largest spacing) is **≥ 3×** the best-resolution axis (smallest spacing), a special rule is triggered; otherwise, the median spacing is adopted as the target.

The computed target spacing is used during the **preprocessing** stage.

## Median Shape (Only for 3D Images)

We crop every image in the dataset with a threshold of 0, record the resulting shapes, and take the median along each dimension. These median values guide the choice of patch size during training and the configuration of the network’s convolution and pooling layers.

## Normalization Scheme

The normalization scheme is determined by the `channel_names` listed in `dataset.yaml`. You can flag a channel to be **skipped** by including a shield keyword (e.g., **label**, **seg**, **mask**) in its name; such channels will not be normalized.

These keywords are **case-insensitive**; simply include them anywhere in the `channel_names`.

All normalization schemes are implemented in [wcode/preprocessing/normalizing.py](../wcode/preprocessing/normalizing.py); only **CT** images and **label** channels receive special handling. If the automatically chosen method is unsuitable, manually edit the corresponding entry in `/Dataset_preprocessed/{DATASET_NAME}/plan.json` **after** dataset analysis but **before** preprocessing.

**Note**
For natural images (RGB or grayscale), we strongly recommend manually switching `ZScoreNormalization` to `GeneralNormalization` in `plan.json`, and—if desired—replacing the RGB-channel `mean` & `std` in `dataset_fingerprint.json` with ImageNet (or another reference dataset) statistics.

# Preprocessing

`Preprocessor` in [wcode/preprocessing/preprocessor.py](../wcode/preprocessing/preprocessor.py) processes the entire dataset by converting every image and label into `.npy` format; accompanying metadata is stored in companion `.pkl` files.

## Cropping (Only for 3D Images)

Zero-threshold cropping usually removes **nothing**; we recommend performing your own **R**egion **o**f **I**nterest (**ROI**) cropping while **converting** the dataset into the supported format.

Code: [wcode/preprocessing/cropping.py](../wcode/preprocessing/cropping.py).

## Normalizing (Both 2/3D Images)

Perform the operations `Preprocessor._normalize` as specified in `plan.json → normalization_schemes`.

In `plan.json`, each entry under `normalization_schemes` corresponds to one modality listed in `dataset.yaml → channel_names`. `dataset_fingerprint.json → foreground_intensity_properties_per_channel`, however, is indexed by **actual channel index**. 

**Therefore**, at runtime, we map the modality-level scheme to every channel that belongs to that modality, combining the normalization plan with the per-modality shape to ensure each channel receives the correct treatment.

Code: [wcode/preprocessing/normalizing.py](../wcode/preprocessing/normalizing.py).

## Resampling (Only for 3D Images)

For 3-D resampling, we handle three cases separately:

1. **Isotropic images** – bicubic interpolation (`order=3`) on all axes.  
2. **Anisotropic images** – linear interpolation (`order=1`) on the low-resolution axis, bicubic (`order=3`) on the remaining axes.  
3. **Labels** – nearest-neighbor interpolation (`order=0`) on every axis.

Code: [wcode/preprocessing/resampling.py](../wcode/preprocessing/resampling.py).
