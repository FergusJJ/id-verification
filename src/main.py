from image_processing import IDCardPipeliner
from ocr import OCRReader


def main():
    pipeliner = IDCardPipeliner()
    image_reader = OCRReader(language="eng", psm=4, oem=3)

    image_path = "./src/pictures/front.jpg"
    pipeliner.process_id_card(image_path)
    if pipeliner.processed_image is None:
        return

    box_coords = image_reader.get_bounding_coords(pipeliner.processed_image)
    if len(box_coords) == 0:
        return
    overlayed_image = pipeliner.draw_bounding_boxes(
        pipeliner.processed_image, box_coords
    )
    pipeliner.show_image_for_debugging(overlayed_image)

    # image_reader.read_fields(pipeliner.processed_image)


if __name__ == "__main__":
    main()
