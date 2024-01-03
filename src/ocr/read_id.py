import pytesseract


class OCRReader:
    def __init__(self, language="eng", psm=3, oem=3):
        self.language = language
        self.psm = psm
        self.oem = oem
        self.config_string = self._build_config_string()
        return

    def _build_config_string(self):
        return f"--oem {self.oem} --psm {self.psm} -l {self.language}"

    def read_fields(self, image):
        print(pytesseract.image_to_string(image, config=self.config_string))

    def get_bounding_coords(self, image):
        # Use Tesseract to get the extended data about elements found in the image
        boxes_str = pytesseract.image_to_data(image)

        # Split the output by lines and then by spaces
        # Note: The first element of the list is the header and should be ignored.
        boxes = [b.split("\t") for b in boxes_str.splitlines()]
        print(f"len boxes: {len(boxes)}")
        # Extract the header row
        header = boxes[0]

        print(header)

        top_idx = header.index("top")
        left_idx = header.index("left")
        width_idx = header.index("width")
        text_idx = header.index("text")
        height_idx = header.index("height")
        conf_idx = header.index("conf")

        box_coords = []

        for box in boxes[1:]:
            if len(box) == len(header):
                print(f"confidence: {box[conf_idx]}")
                if float(box[conf_idx]) != -1 and box[text_idx] != "":
                    x, y, w, h = (
                        int(box[left_idx]),
                        int(box[top_idx]),
                        int(box[width_idx]),
                        int(box[height_idx]),
                    )
                    print(x, y, w, h)
                    box_coords.append(((x, y), (x + w, y + h)))  # diagonal of the rect
        return box_coords
