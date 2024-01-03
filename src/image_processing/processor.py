import cv2 as cv
import numpy as np


class IDCardProcessor:
    def __init__(self, target_height=768, target_width=1024):
        self.target_height = target_height
        self.target_width = target_width

    def _overlay_contours(self, image, contours):
        contour_image = image.copy()
        cv.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
        return contour_image

    @staticmethod
    def load_image(img_path):
        try:
            img = cv.imread(img_path)
            if img is None:
                raise FileNotFoundError(
                    f"Image not found or unable to read: {img_path}"
                )
            return img
        except Exception as e:
            print(f"Error reading the image: {e}")
            return None

    def resize_to_target_resolution(self, image):
        height_ratio = self.target_height / image.shape[0]
        width_ratio = self.target_width / image.shape[1]
        resize_ratio = min(width_ratio, height_ratio)
        print(
            f"resize-ratio: {resize_ratio}\nheight: {image.shape[0]}\nwidth: {image.shape[1]}\ntarget height: {self.target_height}\ntarget width: {self.target_width}"
        )
        new_width = int(image.shape[1] * resize_ratio)
        new_height = int(image.shape[0] * resize_ratio)
        resized_image = cv.resize(
            image, (new_width, new_height), interpolation=cv.INTER_AREA
        )
        return resized_image

    @staticmethod
    def convert_to_grayscale(image):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    @staticmethod
    def apply_blur(image):
        return cv.GaussianBlur(image, (3, 3), 0)

    @staticmethod
    def apply_threshold(image):
        _, thresholded = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return thresholded

    @staticmethod
    def opening(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

    @staticmethod
    def find_contours(image):
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return contours

    def filter_id_card_contours(self, contours, image):
        img_area = image.shape[0] * image.shape[1]
        img_contours = [
            cnt
            for cnt in contours
            if 0.05 * img_area < cv.contourArea(cnt) < 0.90 * img_area
        ]
        return img_contours

    def cut_image_size(self, image):
        grayscale_image = self.convert_to_grayscale(image)
        noise_reduced_image = self.apply_blur(grayscale_image)
        thresholded_image = self.apply_threshold(noise_reduced_image)
        contours = self.find_contours(thresholded_image)
        if contours:
            max_contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(max_contour)
            if 0.05 * grayscale_image.shape[0] * grayscale_image.shape[
                1
            ] < cv.contourArea(max_contour):
                id_image = image[y : y + h, x : x + w]
                resized_image = self.resize_to_target_resolution(id_image)
                return resized_image
            else:
                print("no contours of expected size found")
                return None
        else:
            print("no contours found")
            return None

    def clean_image(self, image):
        image = self.convert_to_grayscale(image)
        image = cv.fastNlMeansDenoising(image, None, 10, 7, 21)
        image = self.apply_blur(image)

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
        image = clahe.apply(image)

        _, image = cv.threshold(
            image, thresh=165, maxval=255, type=cv.THRESH_TRUNC + cv.THRESH_OTSU
        )

        return image

    def ocr_format_image(self, image):
        cleaned_image = self.clean_image(image)
        return cleaned_image

    def preprocess_image(self, image):
        resized_image = self.cut_image_size(image)
        ocr_formatted_image = self.ocr_format_image(resized_image)

        return ocr_formatted_image

    def display_and_save(
        self, image, window_name="Display Window", file_name="id_card.png"
    ):
        cv.imshow(window_name, image)
        while True:
            k = cv.waitKey(1)
            if k == ord("s"):
                cv.imwrite(file_name, image)
            if (
                k == ord("q")
                or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1
            ):
                break
        cv.destroyAllWindows()
