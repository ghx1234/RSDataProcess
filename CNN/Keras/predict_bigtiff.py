import os
import sys
import numpy as np
from PIL import Image
import gdal
import rasterio
import tensorflow as tf
from model import *


def z_score(x):
    return (x - np.mean(x)) / np.std(x, ddof=1)


def read_img(path):
    with rasterio.open(path) as ds:
        data = ds.read(
            out_shape=(ds.count, 512, 512),
            # resampling=Resampling.bilinear
            )
        data = z_score(data)
        img = np.transpose(data, (1, 2, 0))
                # (batch,  H, W, B)
    return img


def predict(image):
    # resize to max dimension of images from training dataset
    w, h, _ = image.shape

    res = model.predict(np.expand_dims(image, 0))
    sess =tf.Session()
    activate = tf.nn.log_softmax(res)
    res = sess.run(activate)
    labels = np.argmax(np.array(res).squeeze(), -1)

    # remove padding and resize back to original image
    labels = np.array(Image.fromarray(labels.astype('uint8')))
    # labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
    return labels


if __name__=='__main__':
    in_folder = '/home/zhoudengji/ghx/code/predict/img'
    out_folder = '/home/zhoudengji/ghx/code/predict/predict/WO'
    model = mwen(pretrained_weights=None, input_shape=(None, 512, 512, 4), class_num=2)
    model.load_weights('/home/zhoudengji/ghx/code/unet-master/womtfe/weights-04-0.98.hdf5')

    for name in os.listdir(in_folder):
        img_path = os.path.join(in_folder, name)

        ds = gdal.Open(img_path)
        result = gdal.GetDriverByName("GTiff").Create(os.path.join(out_folder, name),
                int(np.floor(ds.RasterXSize / 512)) * 512, int(np.floor(ds.RasterYSize / 512)) * 512, 1, gdal.GDT_Byte)
2

###这里最好不要像这样直接一块一块的取，需要去掉带边缘信息的像素

        for x in range(int(np.floor(ds.RasterXSize / 512))):
            for y in range(int(np.floor(ds.RasterYSize / 512))):
                img = ds.ReadAsArray(x * 512, y * 512, 512, 512)
                img = z_score(img)
                img = np.transpose(img, (1, 2, 0))
                img = predict(img)
                result.GetRasterBand(1).WriteArray(img, xoff=x * 512, yoff=y * 512)
        result.SetGeoTransform(ds.GetGeoTransform())  # 写入仿射变换参数
        result.SetProjection(ds.GetProjection())  # 写入投影