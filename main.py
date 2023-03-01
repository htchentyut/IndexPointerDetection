# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os

import cv2

import numpy as np

import matplotlib.pyplot as plt

from graduationDetection import dynamometer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    imgpath = "/media/hchen/T1/data/P19/image/1_19_000_ZL20220927_09_M_4_66cm_V0_H37/"

    imgfiles = os.listdir(imgpath)

    imgfiles.sort()

    img = cv2.imread(imgpath + imgfiles[1500])

    box = np.array([1300, 400, 300, 800])

    dm = dynamometer()

    dm.pointerBaseDetection(img=img, box=box)

    dmimg = img.copy()

    cv2.rectangle(dmimg, tuple(box[:2]), tuple(box[:2] + box[2:]), (0, 255, 0), thickness=5)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(dmimg[:, :, ::-1])
    plt.subplot(1, 2, 2)
    plt.imshow(dm.pointer_img[:, :, ::-1])
    plt.show()

    imgpath = "/media/hchen/T1/data/P19/image/1_19_000_ZL20220927_01_M_4_66cm_V0_H37/"

    imgfiles = os.listdir(imgpath)

    imgfiles.sort()

    print(imgfiles[1650])

    img = cv2.imread(imgpath + imgfiles[1650])

    box = np.array([1020, 450, 380, 850])

    dm = dynamometer()

    dm.slotBaseDetection(img=img, box=box)

    dmimg = img.copy()

    cv2.rectangle(dmimg, tuple(box[:2]), tuple(box[:2] + box[2:]), (0, 255, 0), thickness=5)

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(dmimg[:, :, ::-1])
    plt.subplot(1, 2, 2)
    plt.imshow(dm.pointer_img[:, :, ::-1])
    plt.show()



    # plt.imshow(img[:, :, ::-1])
    #
    # plt.show()
    #
    # dynimg = cv2.cvtColor(img[400:1200, 1300:1600, :], cv2.COLOR_BGR2GRAY)
    #
    # kernel = np.ones((3, 3), np.uint8)
    #
    # ret, thresh = cv2.threshold(dynimg, 200, 255, cv2.THRESH_BINARY)
    #
    # erodethresh = cv2.erode(thresh, kernel, iterations=7)
    #
    # dilatethresh = cv2.dilate(erodethresh, kernel, iterations=11)
    #
    # # plt.figure(figsize=(10, 10))
    # # plt.subplot(1, 4, 1)
    # # plt.imshow(dynimg)
    # # plt.subplot(1, 4, 2)
    # # plt.imshow(thresh)
    # # plt.subplot(1, 4, 3)
    # # plt.imshow(dilatethresh)
    # contours, hierarchy = cv2.findContours(dilatethresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # segimg = img.copy()
    # if contours:
    #     for c in contours:
    #         rect = cv2.minAreaRect(c)
    #         if (rect[0][0] - 300 / 2) < 30:
    #             box = cv2.boxPoints(rect)
    #             box = np.int32(box)
    #             box[:, 0] += 1300
    #             box[:, 1] += 400
    #             cv2.drawContours(segimg, [box], 0, (0, 255, 0), 3)
    #             plt.imshow(segimg[:, :, ::-1])
    #             plt.show()


