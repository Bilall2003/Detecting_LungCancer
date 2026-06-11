import cv2 as cv
import matplotlib.pyplot as plt

def func():

    img = cv.imread("asset/LungCancerCTscan.jpg", cv.IMREAD_GRAYSCALE)
    
    fig,ax = plt.subplots(1, 2, figsize=(40, 15))

    ax[0].imshow(img, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("CT Scan")

    ax[1].hist(img.ravel(), bins=256)
    ax[1].set_title("Histogram")
    
    smooth_img = cv.medianBlur(img,80)


    return fig,img,smooth_img