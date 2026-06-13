import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import rectangle, opening, erosion, dilation
from skimage.measure import regionprops, label
from sklearn.svm import SVC
from skimage import exposure 

img = cv2.imread('lungimg.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title('Original Lung CT Image')

print(img.shape)
print(img.dtype)

plt.hist(img.ravel(), bins=256, range=(0, 256), color ='gray')
plt.title('Histogram of original image')


smooth_img = cv2.medianBlur(img, 3, cv2.IMREAD_GRAYSCALE)
plt.imshow(smooth_img, cmap='gray')

exp = exposure.adjust_gamma(smooth_img, gamma=2)
plt.imshow(exp, cmap='gray')

plt.hist(exp.ravel(), bins=256, range=(0, 256), color ='gray')
plt.title('Histogram of exposure image')


# %%
thresh = threshold_otsu(exp)
binary = exp > thresh

plt.imshow(binary, cmap='gray')

se = rectangle(40, 30)

opened = opening(binary, se)
plt.imshow(opened, cmap='gray')
plt.title('After Opening (Lung Mask)')

eroded = erosion(binary, se)
plt.imshow(eroded, cmap='gray')
plt.title('After Erosion')

dilated = dilation(eroded, se)
plt.imshow(dilated, cmap='gray')
plt.title('After Dilation (Extracted Tumor Region)')

# Label connected regions in the binary image
labeled = label(dilated)
props = regionprops(labeled)

# Extract features from the largest region (tumor)
if props:
    region = max(props, key=lambda r: r.area)
    area = region.area
    perimeter = region.perimeter
    eccentricity = region.eccentricity

    print(f"Area:        {area}")
    print(f"Perimeter:   {perimeter:.4f}")
    print(f"Eccentricity:{eccentricity:.4f}")

    # SVM Classification (using sample data as in assignment)
    # Training data: [area, eccentricity, perimeter] → label (0=normal, 1=cancerous)
    X_train = np.array([
        [1793, 0.7319, 161.698],   # cancerous sample from assignment
        [500,  0.3,    80.0],      # normal sample (example)
    ])
    y_train = np.array([1, 0])  # 1=cancerous, 0=normal

    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    sample = np.array([[area, eccentricity, perimeter]])
    prediction = clf.predict(sample)
    result = "Cancerous (Abnormal)" if prediction[0] == 1 else "Normal"
    print(f"\nSVM Classification Result: {result}")
else:
    print("No regions detected.")


