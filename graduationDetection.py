import cv2
import matplotlib.pyplot as plt
import numpy as np


class dynamometer(object):
    def __init__(self):
        self.img = None
        self.xmin = 0
        self.ymin = 0
        self.width = 0
        self.height = 0
        self.pointer = 0
        self.pointer_left = 0
        self.pointer_right = 0
        self.recognition = 0
        self.center_thresh = 60
        self.pointer_img = None

    def pointerBaseDetection(self, img=None, box=None):
        if img.any():
            if not box.any():
                self.img = cv2.cvtColor(img[400:1200, 1300:1600, :], cv2.COLOR_BGR2GRAY)
            else:
                self.xmin = box[0]
                self.ymin = box[1]
                self.width = box[2]
                self.height = box[3]
                self.img = cv2.cvtColor(img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :], cv2.COLOR_BGR2GRAY)

            kernel = np.ones((3, 3), np.uint8)

            ret, thresh = cv2.threshold(self.img, 200, 255, cv2.THRESH_BINARY)

            erode_thresh = cv2.erode(thresh, kernel, iterations=7)

            dilate_thresh = cv2.dilate(erode_thresh, kernel, iterations=11)

            contours, hierarchy = cv2.findContours(dilate_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            segimg = img.copy()

            if contours:
                for c in contours:
                    rect = cv2.minAreaRect(c)
                    if np.abs(rect[0][0] - self.width / 2) < self.center_thresh:
                        pointer_box = np.int32(cv2.boxPoints(rect))
                        pointer_box[:, 0] += self.xmin
                        pointer_box[:, 1] += self.ymin
                        cv2.drawContours(segimg, [pointer_box], 0, (0, 0, 255), 5)
                        # plt.imshow(segimg[:, :, ::-1])
                        # plt.show()
        self.pointer_img = segimg.copy()

    def slotBaseDetection(self, img=None, box=None):
        if img.any():
            if not box.any():
                self.img = cv2.cvtColor(img[450:1200, 1000:1450, :], cv2.COLOR_BGR2GRAY)
            else:
                self.xmin = box[0]
                self.ymin = box[1]
                self.width = box[2]
                self.height = box[3]
                self.img = cv2.cvtColor(img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :], cv2.COLOR_BGR2GRAY)

        kernel = np.ones((5, 5), np.uint8)

        ret, thresh = cv2.threshold(self.img, 128, 255, cv2.THRESH_BINARY_INV)

        erode_thresh = cv2.erode(thresh, kernel, iterations=3)

        dilate_thresh = cv2.dilate(erode_thresh, kernel, iterations=7)

        contours, hierarchy = cv2.findContours(dilate_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segimg = img.copy()

        if contours:
            for c in contours:
                rect = cv2.minAreaRect(c)

                if np.abs(rect[0][0] - self.width / 2) < self.center_thresh:
                    pointer_box = np.int32(cv2.boxPoints(rect))
                    pointer_box[:, 0] += self.xmin
                    pointer_box[:, 1] += self.ymin
                    cv2.drawContours(segimg, [pointer_box], 0, (0, 0, 255), 5)
                    # plt.imshow(segimg[:, :, ::-1])
        self.pointer_img = segimg.copy()
