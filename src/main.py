from image_processing import IDCardProcessor


def main():
    processor = IDCardProcessor()
    image_path = "./src/pictures/front.jpg"
    id_info = processor.process_id_card(image_path)
    print(id_info)


if __name__ == "__main__":
    main()
