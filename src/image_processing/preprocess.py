import cv2 as cv
import numpy as np


def read_img(img_path: str):
    try:
        img = cv.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unable to read: {img_path}")
        return img

    except Exception as e:
        print(f"Error reading the image: {e}")
        return None


def resize_img(image: cv.Mat, scale_percent: int):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(image, dim, interpolation=cv.INTER_AREA)


def preprocess_img(image):
    gs_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh_image = cv.adaptiveThreshold(
        gs_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
    )
    contours, _ = cv.findContours(
        thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    if contours:
        id_contour = max(contours, key=cv.contourArea)

        # Create a mask for the ID card
        mask = np.zeros_like(gs_image)
        cv.drawContours(mask, [id_contour], -1, color=255, thickness=cv.FILLED)

        # Apply the mask to isolate the ID card, ensure the mask is three-channel
        mask_3d = cv.merge([mask, mask, mask])
        id_card = cv.bitwise_and(image, mask_3d)
        return id_card
    else:
        return image  # Return the original image if no contours are found


def pipeline(image):
    img = resize_img(image, 10)
    img = preprocess_img(img)
    cv.imshow("Display window", img)
    while True:
        k = cv.waitKey(1)

        if k == ord("s"):  # Save on 's' key press
            cv.imwrite("rewrite.png", img)

        if (
            k == ord("q")
            or cv.getWindowProperty("Display window", cv.WND_PROP_VISIBLE) < 1
        ):
            break

    cv.destroyAllWindows()


img = read_img("../../../Pictures/front.jpg")  # "/home/fergus/Pictures/Screenshots/")
if img is None:
    exit(1)
pipeline(img)
