from skimage.segmentation import slic, mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np


def SuperPixel_by_SLIC(img_path, save_path, n_segments=500, compactness=20):
    img = io.imread(img_path)
    segments = slic(
        img,
        n_segments=n_segments,
        compactness=compactness,
        enforce_connectivity=True,
        convert2lab=True,
    )
    out = mark_boundaries(np.zeros_like(img), segments)
    io.imsave(save_path, out.astype(np.uint8) * 255)


if __name__ == "__main__":
    SuperPixel_by_SLIC(
        "./Dataset/MoNuSegFully/imagesTr/MoNuSegFully_0000_0000.png",
        "./test_0.jpg",
        n_segments=1000,
        compactness=20,
    )
