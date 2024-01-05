from image_processing import IDCardPipeliner
from ocr import OCRReader, run_clustering


def main():
    pipeliner = IDCardPipeliner()
    image_reader = OCRReader(language="eng", psm=4, oem=3)

    image_path = "./src/pictures/front.jpg"
    pipeliner.process_id_card(image_path)
    if pipeliner.processed_image is None:
        return
    pipeliner.show_image_for_debugging(pipeliner.resized_image)
    box_coords = image_reader.get_bounding_coords(pipeliner.processed_image)
    if len(box_coords) == 0:
        return
    clustered = run_clustering(box_coords)
    points_image = pipeliner._draw_points(pipeliner.resized_image, clustered)
    pipeliner.show_image_for_debugging(points_image)
    image_reader.read_lines(pipeliner.processed_image)


if __name__ == "__main__":
    main()
