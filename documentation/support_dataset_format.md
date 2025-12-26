# Supported Structures of the Dataset Folder

You should write your own `convert_··.py` scripts to make the file organization format meet the requirements. We give some examples in [wcode/convert_datasets](../wcode/convert_datasets).

## File Name

**Image** file names follow the pattern:  
`<custom_name>_<case_id>_<modality_id><extension>`.  

**Label** file names follow the pattern:  
`<custom_name>_<case_id><extension>`.  

Images and labels are matched by `case_id`; the file extension must be consistent across the entire dataset.

## With Official Split

```
Dataset
├── LNQ2023
|   ├── imagesTr
|   |   ├── LNQ2023_0005_0000.nii.gz
|   |   ├── ...
|   |
|   ├── (imagesTs)
|   ├── imagesVal
|   ├── labelsTr
|   |   ├── LNQ2023_0005.nii.gz
|   |   ├── ...
|   |
|   ├── (labelsTs)
|   ├── labelsVal
|   └── dataset.yaml
├── ...
```

For datasets that already provide **official train/validation/(test) splits**, we recommend reorganizing the raw dataset into the above directory format.

**Note**

- If the official release does not provide an explicit validation split, we treat the test set as the validation set.  
  In this case, the `imagesTs/` and `labelsTs/` folders **can be omitted**, and the officially-released “test” data should be placed under `imagesVal/` and `labelsVal/`.
- If you prefer the conventional machine learning split (train / val / test), we recommend creating your own validation set while converting the dataset and moving the corresponding images and labels into the respective folders (original train set -> train + val set ->  `imagesTr/`, `imagesVal/`, `labelsTr/` and `labelsVal/`, original test set -> `imagesTs` and `labelsTs`).

## Without Official Split

For datasets **without official splits**, we recommend organizing the files in the following structure:

```
Dataset
├── CTLymphNodes
|   ├── images
|   |   ├── CTLymphNodes_0001_0000.nii.gz
|   |   ├── ...
|   |
|   ├── labels
|   |   ├── CTLymphNodes_0001.nii.gz
|   |   ├── ...
|   |
|   └── dataset.yaml
├── ...
```

In this scenario, the split is performed during dataset parsing and preprocessing. You can choose either a **train / val / test** split or **k-fold cross-validation**.

## dataset.yaml

`dataset.yaml` stores dataset-level metadata:

* **modality/channels** (channel_names) – used during analysis and preprocessing; e.g., normalization strategies are automatically selected by modality name.  
* **file_extension** (files_ending) – required by the data loader.  
* **class_labels** (labels) – maps semantic names to label indices for consistent encoding in preprocessing, inference, and evaluation (e.g., map natural-image categories in order).

```
# 1. medical images
channel_names:
  '0': T1w_MRI
  '1': T2w_MRI
files_ending: .nii.gz
labels:
  background: 0
  GTVp: 1
  GTVnd: 2

# 2. natural image (if you already mapped the label yourself in the conver convert_··.py)
channel_names:
  '0': RGB
  '1': Depth image
files_ending: .png
labels:
  background: 0
  book: 1
  phone: 2
  
# 3. natural image (if you do not map the label yourself in the conver convert_··.py)
channel_names:
  '0': RGB
  '1': Depth image
files_ending: .png
labels:
  background: [0, 0, 0]
  book: [255, 0, 0]
  phone: [0, 255, 0]
```

**Note**

- All channels/modalities must share the same spatial size and be pixel-wise aligned.  
- For natural images, we treat RGB *et al.* (3 channels) as a single modality, but save it as 3 channels.  
  Grayscale or depth images, even if stored as 3-channel files, are verified channel-by-channel; if the channels are identical, they are collapsed to a single channel during preprocessing.
- The values of label in the third situation are [0, 0, 0] -> 0, [255, 0, 0] -> 1, [0, 255, 0] -> 2.

