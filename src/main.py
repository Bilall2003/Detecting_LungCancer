import cv2 as cv
import matplotlib.pyplot as plt

def func():
    
    img=cv.imread("asset/LungCancerCTscan.jpg")
    
    img_rgb=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    fig,ax=plt.subplots()
    ax.imshow(img_rgb)
    ax.axis("off")
    
    return fig
    