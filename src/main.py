import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def func(image):

    # =========================
    # 1. Load image (SAFE)
    # =========================
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found at: {image}")

    # =========================
    # 2. Median Blur (denoise)
    # =========================
    smooth_img = cv.medianBlur(img, 3)

    # =========================
    # 3. CLAHE (contrast enhancement)
    # =========================
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(smooth_img)

    # =========================
    # 4. Otsu Thresholding (Segmentation)
    # =========================
    _, mask = cv.threshold(
        enhanced, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    # =========================
    # 5. Morphological Cleaning
    # =========================
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    cleaned = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    cleaned = cv.dilate(cleaned, kernel, iterations=1)

    # =========================
    # 6. Feature Extraction
    # =========================
    area = cv.countNonZero(cleaned)

    contours, _ = cv.findContours(
        cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    perimeter = 0
    eccentricity = 0

    if contours:
        cnt = max(contours, key=cv.contourArea)

        perimeter = cv.arcLength(cnt, True)

        if len(cnt) >= 5:
            try:
                (x, y), (MA, ma), angle = cv.fitEllipse(cnt)

                if ma != 0:
                    eccentricity = np.sqrt(1 - (MA / ma) ** 2)

            except:
                eccentricity = 0

    # =========================
    # 7. Visualization
    # =========================
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    ax[0, 0].imshow(img, cmap="gray")
    ax[0, 0].set_title("Original")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(smooth_img, cmap="gray")
    ax[0, 1].set_title("Median Blur")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(enhanced, cmap="gray")
    ax[0, 2].set_title("CLAHE Enhanced")
    ax[0, 2].axis("off")

    ax[1, 0].imshow(mask, cmap="gray")
    ax[1, 0].set_title("Otsu Mask")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(cleaned, cmap="gray")
    ax[1, 1].set_title("Morphology Cleaned")
    ax[1, 1].axis("off")

    ax[1, 2].hist(img.ravel(), bins=256, range=(0, 256))
    ax[1, 2].set_title("Histogram")

    plt.tight_layout()

    # =========================
    # 8. Return results
    # =========================
    return fig, img, smooth_img, enhanced, cleaned, area, perimeter, eccentricity