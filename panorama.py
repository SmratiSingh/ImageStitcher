import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def findPoints(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    return kp, des


def matchFeatures(des1, des2, img1, img2):
    topmatches = list()
    idx1 = 0
    for i in des1:
        allDis = list()
        idx2 = 0
        for j in des2:
            d = cv2.norm(i, j)
            item = [d, idx1, idx2]
            idx2 += 1
            allDis.append(item)
        idx1 += 1
        allDis.sort()
        topmatches.append(allDis[0:2])

    return topmatches


def goodMatches(matches):
    good = []
    for m, n in matches:
        # print("m[0]= ", m[0], " ,n[0]= ", n[0])
        if m[0] < 0.5 * n[0]:
            good.append(m)

    return good


def findHomography(randFour):
    homoList = list()

    # print("randFour values = ", randFour)

    for pt in randFour:
        # print("pt.item(0) ", pt.item(0))
        xVal = [-pt.item(0), -pt.item(1), -1, 0, 0, 0, pt.item(2) * pt.item(0), pt.item(2) * pt.item(1), pt.item(2)]
        yVal = [0, 0, 0, -pt.item(0), -pt.item(1), -1, pt.item(3) * pt.item(0), pt.item(3) * pt.item(1), pt.item(3)]
        homoList.append(xVal)
        homoList.append(yVal)

    homoMat = np.matrix(homoList)

    u, s, v = np.linalg.svd(homoMat)

    h = np.reshape(v[8], (3, 3))
    h = (1/h.item(8)) * h

    return h


def calcDist(i, homo):
    p1 = np.transpose(np.matrix([i[0].item(0), i[0].item(1), 1]))
    estimatep2 = np.dot(homo, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2

    p2 = np.transpose(np.matrix([i[0].item(2), i[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


def ransacAlgo(coorMat):

    maxInlier = []
    flag =0
    for j in range(0, 1000):
        randFour = []
        for i in range(0, 4):
            randmatch = random.choice(coorMat)
            # if flag == 0:
            #     print("Randmatch = ", randmatch)
            randFour.append(randmatch)
            # print("randFour 1111: ", randFour)
        # if flag == 0:
        #     print("randFour after all appending: ", randFour)
        # flag = 1
        homo = findHomography(randFour)
        # print("Homography function output: ", homo)
        inlier = list()
        for i in coorMat:
            dist = calcDist(i, homo)

            if dist < 5:
                inlier.append(i)

        if len(inlier) > len(maxInlier):
            maxInlier = inlier
            H = homo
    # print("Final H in function = ", H)
    # print("H size: ", H.shape)
    # print("H.item(0)= ", H.item(0))
    return maxInlier, H


def main(inputPath1, inputPath2, outputPath):
    """...........Reading images................."""
    img_r = cv2.imread(inputPath2)
    # cv2.imshow('Original left', img_)
    # img_r = cv2.resize(img_, (0, 0), fx=0.15, fy=0.15)
    # cv2.imshow('Resized left', img_r)
    img1 = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    imgr = cv2.imread(inputPath1)
    # cv2.imshow('Original right', img)
    # imgr = cv2.resize(img, (0, 0), fx=0.15, fy=0.15)
    # cv2.imshow('Resized right', imgr)
    img2 = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
    print("read")
    """............Detecting keypoints and descriptors............."""

    kp1, des1 = findPoints(img1)
    kp2, des2 = findPoints(img2)
    print("keypoint detected")
    """............Matching keypoints in both images................"""

    matches = matchFeatures(des1, des2, img_r, imgr)
    print("matching done")
    """........Finding good matches.............."""
    good = goodMatches(matches)
    print("Length of good = ", len(good))
    # print("Good = ", good)

    MIN_MATCH_COUNT = 30
    if len(good) > MIN_MATCH_COUNT:
        coordList = list()
        for m in good:
            (x1, y1) = kp1[m[1]].pt
            (x2, y2) = kp2[m[2]].pt
            coordList.append([x1, y1, x2, y2])
        # print("coordlist[1] = ", coordList[1])
        coordMat = np.matrix(coordList)
        # print("coordMat = ", coordMat[0])
        maxInlier, H = ransacAlgo(coordMat)
        print("Homography matrix H = ", H)
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        cv2.imshow("original_image_overlapping.jpg", img2)
        # cv2.waitKey(0)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

    # Warping
    dst = cv2.warpPerspective(img_r, H, (img_r.shape[1] + imgr.shape[1], imgr.shape[0]))
    # cv2.imshow("warp_result.jpg", dst)

    print("imrshape", imgr.shape)

    dst[0:imgr.shape[0], 0:imgr.shape[1]] = imgr

    # dst[0:img.shape[0], 0+img_.shape[1]:img_.shape[1]+img.shape[1]] = img
    # cv2.imshow("adding_left_side.jpg", dst)

    def trim(frame, countTrim):
        countTrim += 1
        print("entering trim")
        # crop top
        if not np.sum(frame[0]):
            print("top crop")
            return trim(frame[1:], countTrim)
        # crop bottom
        elif not np.sum(frame[-1]):
            print("bottom crop")
            return trim(frame[:-2], countTrim)
        # crop left
        elif not np.sum(frame[:, 0]):
            print("left crop")
            return trim(frame[:, 1:], countTrim)
        # crop right
        elif not np.sum(frame[:, -1]):
            print("right crop")
            return trim(frame[:, :-65], countTrim)
        return frame, countTrim

    print("img1 right: ", img_r.shape)
    print("exiting trim")
    print("img2 left: ", imgr.shape)
    print("pan: ", dst.shape)
    countTrim = 0
    output, count = trim(dst, countTrim)
    cv2.imshow("original_image_stiched_crop.jpg", output)
    # cv2.waitKey(0)
    cv2.imwrite(outputPath, output)
    print("Count of trim call: ", count)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    INPUT_DIR = "./inputs/"
    OUTPUT_DIR = "./outputs/"
    INTERIM_DIR = "./interim/"

    main(INPUT_DIR + "ub2.jpg", INPUT_DIR + "ub1.jpg", INTERIM_DIR + "interimPan01.jpg")
    main(INPUT_DIR + "ub3.jpg", INPUT_DIR + "ub2.jpg", INTERIM_DIR + "interimPan02.jpg")
    main(INTERIM_DIR + "interimPan01.jpg", INTERIM_DIR + "interimPan02.jpg", OUTPUT_DIR + "finalPan.jpg")



