import cv2
import numpy as np
import matplotlib.pyplot as plt
name = 'demo1.jpg' #檔案名稱
def gray_equalize_hist(img):
    # 存下原圖的數據
    img_histogram = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_histogram[img[i, j, 0]] += 1
    # 計算累積直方圖並計算分布位置
    total = img.shape[0] * img.shape[1]
    cur = 0
    new_val = np.zeros(256)
    map = {}
    for i in range(256):
        cur += img_histogram[i]
        new_val[int(cur / total * 255)] += img_histogram[i]
        map[i] = int(cur / total * 255)
    # 印出原圖的直方圖
    plt.figure()
    plt.bar(range(256), img_histogram)
    plt.title(f'{name} gray Original Histogram')
    plt.savefig(f'{name} gray Original_Histogram.png')
    # 印出均衡化後的直方圖
    plt.figure()
    plt.bar(range(256), new_val)
    plt.title(f'{name} gray Equalized Histogram')
    plt.savefig(f'{name} gray Equalized_Histogram.png')
    # return 均衡化後的圖片
    new_img = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i, j, 0] = map[img[i, j, 0]]
    return new_img
def rgb_equalize_hist(img):
    # 存下原圖rgb三層的分別數據
    img_histogram = np.zeros((256, 3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                img_histogram[img[i, j, k], k] += 1
    # 計算累積直方圖並計算分布位置
    total = img.shape[0] * img.shape[1]
    cur = np.zeros(3)
    new_val = np.zeros((256, 3))
    map = [{} for _ in range(3)]
    for i in range(256):
        for k in range(3):
            cur[k] += img_histogram[i, k]
            new_val[int(cur[k] / total * 255), k] += img_histogram[i, k]
            map[k][i] = int(cur[k] / total * 255)
    # 分別印出每個通道的原圖直方圖
    channel_names = ['Blue', 'Green', 'Red']
    for k in range(3):
        plt.figure()
        plt.bar(range(256), img_histogram[:, k],label=channel_names[k])
        plt.title(f'{name} BGR Original Histogram - {channel_names[k]} Channel')
        plt.legend()
        plt.savefig(f'{name} BGR Original_Histogram_{channel_names[k]}.png')
    # 分別印出每個通道的均衡化後的直方圖
    for k in range(3):
        plt.figure()
        plt.bar(range(256), new_val[:, k],label=channel_names[k])
        plt.title(f'{name} BGR Equalized Histogram - {channel_names[k]} Channel')
        plt.legend()
        plt.savefig(f'{name} BGR Equalized_Histogram_{channel_names[k]}.png')
    # return 均衡化後的圖片
    new_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                new_img[i, j, k] = map[k][img[i, j, k]]
    return new_img

def hsv_equalize_hist(img):
    # 將圖像從 BGR 轉換為 HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 計算 V 通道的直方圖
    img_histogram = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_histogram[img[i, j, 2]] += 1
    # 計算累積直方圖與分布位置
    total = img.shape[0] * img.shape[1]
    cur = 0
    new_val = np.zeros(256)
    map = {}
    for i in range(256):
        cur += img_histogram[i]
        new_val[int(cur / total * 255)] += img_histogram[i]
        map[i] = int(cur / total * 255)
    # 印出原圖的直方圖
    plt.figure()
    plt.bar(range(256), img_histogram)
    plt.title(f'{name} HSV Original Histogram')
    plt.savefig(f'{name} HSV Original_Histogram.png')
    # 印出均衡化後的直方圖
    plt.figure()
    plt.bar(range(256), new_val)
    plt.title(f'{name} HSV Equalized Histogram')
    plt.savefig(f'{name} HSV Equalized_Histogram.png')
    # 創建新圖像，並應用直方圖均衡化
    new_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i, j, 0] = img[i, j, 0]
            new_img[i, j, 1] = img[i, j, 1]
            new_img[i, j, 2] = map[img[i, j, 2]]
    # 將圖像從 HSV 轉換回 BGR 並return
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
    return new_img

if __name__ == "__main__":
    image = cv2.imread(f'{name}',cv2.IMREAD_UNCHANGED)    
    if len(image.shape) == 2:
        # 灰階圖像從[x, y]轉成[x, y, 1]
        image.shape = image.shape + (1,)
    if len(image.shape) == 3 and image.shape[2] == 1:
        # 灰階圖像
        image = gray_equalize_hist(image)
        cv2.imwrite(f'{name}_gray_equalized.jpg', image)
    elif len(image.shape) == 3 and image.shape[2] >= 3:
        # 彩色圖像透過BGR進行均化
        image1 = rgb_equalize_hist(image)
        cv2.imwrite(f'{name}_rgb_equalized.jpg', image1)
        # 彩色圖透過HSV進行均化
        image2 = hsv_equalize_hist(image)
        cv2.imwrite(f'{name}_hsv_equalized.jpg', image2)
    