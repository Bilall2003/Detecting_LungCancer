import cv2 as cv
import matplotlib.pyplot as plt

def func():
    
    img=cv.imread("asset/LungCancerCTscan.jpg", cv.IMREAD_GRAYSCALE)
    
    # img_rgb=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    fig,ax=plt.subplots()
    ax.imshow(img,cmap="gray")
    ax.axis("off")
    
    return fig
    
    
    