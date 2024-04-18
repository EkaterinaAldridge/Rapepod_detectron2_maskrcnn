from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


# 定义一个中点函数，后面会用到
def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


ignore_size = 10000
coin_real_area = 491
coin_real_length = 78.54


def read_img(filename):
    img = cv2.imread(filename, 1)
    # dst1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def gaussian_blur(img):
    gaussian_img = cv2.GaussianBlur(img, (5, 5), 10)
    return gaussian_img


def gray_procession(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def threshold_procession(img, threshold):
    _, threshold_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return threshold_img


def draw_shape(img1, img2):
    contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 初始化 'pixels per metric'
    pixelsPerMetric = None
    areaPixelsMetric = None
    # cv2.drawContours(img2, contours, -1, (0, 0, 255), 3)
    # cv2.namedWindow('final_img', 1)
    # cv2.imshow('final_img', img2)
    # cv2.waitKey()
    for c in contours:
        # 如果当前轮廓的面积太少，认为可能是噪声，直接忽略掉
        if cv2.contourArea(c) < ignore_size:
            continue
        # 根据物体轮廓计算出外切矩形框

        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # 按照top-left, top-right, bottom-right, bottom-left的顺序对轮廓点进行排序，并绘制外切的BB，用绿色的线来表示
        box = perspective.order_points(box)

        # 绘制周长轮廓
        cv2.drawContours(img2, c, -1, (0, 255, 0), 3)

        # 分别计算top-left 和top-right的中心点和bottom-left 和bottom-right的中心点坐标
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # 分别计算top-left和top-right的中心点和top-right和bottom-right的中心点坐标
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # 初始化测量指标值，参考物体在图片中的宽度已经通过欧氏距离计算得到，参考物体的实际大小已知
        if pixelsPerMetric is None:
            pixelsPerMetric = cv2.arcLength(c, True) / coin_real_length
        if areaPixelsMetric is None:
            areaPixelsMetric = cv2.contourArea(c) / coin_real_area
    return pixelsPerMetric, areaPixelsMetric


def getcoin_size(imgname):
    filename = imgname
    img = read_img(filename)
    gaussian_img = gaussian_blur(img)
    gray_img = gray_procession(gaussian_img)
    threshold_img = threshold_procession(gray_img, 110)
    coin_size, coin_area = draw_shape(threshold_img, img)
    return coin_size, coin_area
