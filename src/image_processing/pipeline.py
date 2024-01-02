import os
import cv2 as cv
import pytesseract
from image_processing.extractor import IDCardExtractor


# Class to load the image and preprocess it using the extractor


class IDCardProcessor(IDCardExtractor):
    def __init__(self, target_width=1024, target_height=768):
        super().__init__(target_width=target_width, target_height=target_height)

    @staticmethod
    def show_image_for_debugging(image, window_name="OCR Input"):
        cv.imshow(window_name, image)
        while True:
            # If 'q' is pressed within the waitKey period, break from the loop
            if cv.waitKey(1) & 0xFF == ord("q"):
                print("'q' pressed, closing...")
                break
            # If the 'x' close button is clicked, break from the loop
            if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                print("'x' clicked, closing...")
                break
        cv.destroyAllWindows()
        print("Windows destroyed")

    def read_text_from_image(self, image):
        return pytesseract.image_to_string(image)

    def process_id_card(self, img_path):
        image = self.load_image(img_path)
        if image is None:
            return "Unable to load image."
        id_card = self.preprocess_image(image)
        if id_card is None:
            return "No ID card detected."
        id_text = self.read_text_from_image(id_card)
        self.show_image_for_debugging(id_card)
        return id_text
