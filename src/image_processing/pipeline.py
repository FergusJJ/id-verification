import cv2 as cv
from image_processing.processor import IDCardProcessor

# Class to load the image and preprocess it using the extractor


class IDCardPipeliner(IDCardProcessor):
    def __init__(self, target_width=1024, target_height=768):
        super().__init__(target_width=target_width, target_height=target_height)

    @staticmethod
    def draw_bounding_boxes(base_image, boxes, color=(125, 255, 0), thickness=2):
        """
        Draw bounding boxes on an image.

        :param base_image: The image to draw on.
        :param boxes: A list of tuples with coordinates [(x, y, x+w, y+h), ...].
        :param color: The color of the bounding boxes (B, G, R).
        :param thickness: The thickness of the bounding box edges.
        :return: The image with drawn bounding boxes.
        """
        # Create a copy of the original image to draw the boxes on
        image_with_boxes = base_image.copy()

        # Draw each bounding box
        print(f"boxes: {boxes}")
        for start_point, end_point in boxes:
            # Draw a rectangle on the image using the coordinates
            cv.rectangle(image_with_boxes, start_point, end_point, color, thickness)

        return image_with_boxes

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

    def process_id_card(self, img_path):
        image = self.load_image(img_path)
        if image is None:
            print("Unable to load image.")
            return
        id_card = self.preprocess_image(image)
        if id_card is None:
            print("No ID card detected.")
            return
        self.processed_image = id_card
        # self.show_image_for_debugging(self.processed_image)
        return
