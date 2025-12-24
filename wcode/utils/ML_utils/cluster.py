import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans


def loadData(filePath):
    data = []
    img = image.open(filePath)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel((i, j))
            data.append([x / 256.0, y / 256.0, z / 256.0])
    return np.array(data), m, n


if __name__ == "__main__":
    imgData, row, col = loadData(
        "./Dataset/MoNuSegFully/imagesTr/MoNuSegFully_0000_0000.png"
    )
    print(imgData.shape)
    label = KMeans(n_clusters=3).fit_predict(imgData)
    label = label.reshape([row, col])
    pic_new = image.new("L", (row, col))
    for i in range(row):
        for j in range(col):
            pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
    pic_new.save("result-bull-4.jpg")
