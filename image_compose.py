## fncImgCompose(front_img, back_img, size, pos)
##
## front_image ... charactor
## back_image  ... background
## size        ... size of charactor(x,y)
## pos         ... position of charactor(x,y)

import cv2
import numpy as np

def fncMakeMask(figure):

    # HSV に変換する。
    hsv = cv2.cvtColor(figure, cv2.COLOR_BGR2HSV)

    # 2値化する。
    bin_img = cv2.inRange(hsv, (0, 10, 0), (255, 255, 255))

    # 輪郭抽出する。
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積が最大の輪郭を取得する
    contour = max(contours, key=lambda x: cv2.contourArea(x))

    # マスク画像を作成する。
    mask = np.zeros_like(bin_img)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)

    return mask


def fncImgCompose(front_img, back_img, size, pos):
    
    # front_image ... charactor
    # back_image  ... background
    # size  ... size of charactor(x,y)
    # pos   ... position of charactor(x,y)

    front_img = cv2.resize(front_img,size)
    mask = fncMakeMask(front_img)

    x, y = pos
    
    # 幅、高さは前景画像と背景画像の共通部分をとる
    w = min(front_img.shape[1], back_img.shape[1] - x)
    h = min(front_img.shape[0], back_img.shape[0] - y)

    # 合成する領域
    fg_roi = front_img[:h, :w]               # 前景画像のうち、合成する領域
    bg_roi = back_img[y : y + h, x : x + w]  # 背景画像のうち、合成する領域

    # 合成する。
    bg_roi[:] = np.where(mask[:h, :w, np.newaxis] == 0, bg_roi, fg_roi)

    return back_img


##def main():
##
##    # 画像を読み込む    
##    front_image = cv2.imread(r'rsc/figure_standing.jpg')
##    back_image = cv2.imread(r'rsc/background.png')
##
##    # マージする
##    back_image = fncImgCompose(front_image, back_image, (100,100),(500,500))
##
##    cv2.imshow('result', back_image)
##    cv2.waitKey(0)
##
##if __name__ == '__main__':
##    main()
