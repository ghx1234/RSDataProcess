import cv2
import numpy as np
import gdal
'''channel first numpy'''

class zengqiang:
    def __init__(self, img):
        self.array = img

    def fanzhuan(self, flag): #0 for v;1for h; -1for h and v ; 2for none翻转
        array = []
        if flag < 0:
            for i in range(self.array.shape[0]):
                band = self.array[i]
                band = np.flip(band, 0)
                band = np.flip(band, 1)
                array.append(band)
            self.array = np.array(array)
        elif flag == 2:
            pass
        else:
            for i in range(self.array.shape[0]):
                band = self.array[i]
                band = np.flip(band, flag)
                array.append(band)
            self.array = np.array(array)

    def pingyi(self, x, y): # x和y在[-512, 512]
        array = []
        M = np.float32([[1, 0, x], [0, 1, y]])
        for i in range(self.array.shape[0]):
            band = self.array[i]
            band = cv2.warpAffine(band, M, (band.shape[1],band.shape[0]))
            array.append(band)
        self.array = np.array(array)

    def xuanzhuansuofang(self, r, s): #r[-180, 180], s[0,n]
        array = []
        (h, w) = self.array.shape[1:3]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, r, s)
        for i in range(self.array.shape[0]):
            band = self.array[i]
            band = cv2.warpAffine(band, M, (w, h))
            array.append(band)
        self.array = np.array(array)

    def liangdu(self, r):
        array = self.array
        self.array = array * r

    def standard(self):
        array = self.array
        self.array = (array - np.mean(array)) / np.std(array, ddof=1)


if __name__=='__main__':
    ds = gdal.Open(r'Z:\guohongxiang\project\water\GF\mylabel\clippde\train_val\img\10_1463_6.tif')
    x = ds.ReadAsArray()
    # x = np.transpose(x, (1,2,0))
    x = zengqiang(x)
    x.xuanzhuansuofang(0, 1.2)
    x = x.array
    driver = gdal.GetDriverByName("GTiff")
    result=driver.Create(r"F:\01学习\02陆表水体\lunwen\chutu\zengqiang\fang.tif",512,512,4,gdal.GDT_UInt16)
    for i in range(4):
        result.GetRasterBand(i+1).WriteArray(x[i])
