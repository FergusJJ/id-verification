import pytesseract


def read_fields(image):
    print("reading with pt")
    print(pytesseract.image_to_string(image))
    print("done reading with pt")
    return None
