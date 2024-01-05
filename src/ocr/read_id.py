from collections import defaultdict
import pytesseract
import re


class OCRReader:
    NAME_PATTERN = re.compile(r"^[a-zA-Z]+(?:\s[a-zA-Z]+){1,4}$")
    DOB_PATTERN = re.compile(r"(0[1-9]|[12]\d|3[01])\.(0[1-9]|1[0-2])\.\d{4}")
    ADDRESS_PATTERN = re.compile(r"\d+\s[a-zA-Z\s]+\,\s[A-Z]{1,2}\d{1,2}\s?\d?[A-Z]{2}")
    LICENSE_NUMBER_PATTERN = re.compile(r"[A-Z]{5}\d{6}[A-Z0-9]{5}")
    ISSUE_EXPIRY_DATE_PATTERN = re.compile(r"\d{2}\.\d{2}\.\d{4}")

    def __init__(self, language="eng", psm=3, oem=3):
        self.language = language
        self.psm = psm
        self.oem = oem

        self.config_string = self._build_config_string()
        return

    def _build_config_string(self):
        return f"--oem {self.oem} --psm {self.psm} -l {self.language}"

    def template_match(self, image):
        data = pytesseract.image_to_data(
            image, config=self.config_string, output_type=pytesseract.Output.DICT
        )
        num_boxes = len(data["text"])
        data_values = defaultdict(list)
        for i in range(num_boxes):
            if int(data["conf"][i]) > 50:
                if re.match(OCRReader.NAME_PATTERN, data["text"][i]):
                    data_values["name"].append(data["text"][i])
                elif re.match(OCRReader.DOB_PATTERN, data["text"][i]):
                    data_values["dob"].append(data["text"][i])
                elif re.match(OCRReader.LICENSE_NUMBER_PATTERN, data["text"][i]):
                    data_values["license_number"].append(data["text"][i])
                elif re.match(OCRReader.ADDRESS_PATTERN, data["text"][i]):
                    data_values["address"].append(data["text"][i])
                elif re.match(OCRReader.ISSUE_EXPIRY_DATE_PATTERN, data["text"][i]):
                    data_values["issue_expiry_date"].append(data["text"][i])
                else:
                    print(f"no match for: {data['text'][i]}")
        return

    def read_fields(self, image):
        print(pytesseract.image_to_string(image, config=self.config_string))

    def read_lines(self, image):
        boxes_df = pytesseract.image_to_data(
            image, config=self.config_string, output_type=pytesseract.Output.DATAFRAME
        )
        for line_num, same_line_words in boxes_df.groupby("line_num"):
            same_line_words = same_line_words[same_line_words["conf"] > 60]
            words = same_line_words["text"].values

            line = " ".join(words)
            print(f"{line_num} '{line}'")

    def get_bounding_coords(self, image):
        boxes_str = pytesseract.image_to_data(image)

        # Split the output by lines and then by spaces
        # Note: The first element of the list is the header and should be ignored.
        boxes = [b.split("\t") for b in boxes_str.splitlines()]

        # Extract the header row

        header = boxes[0]
        top_idx = header.index("top")
        left_idx = header.index("left")
        width_idx = header.index("width")
        text_idx = header.index("text")
        height_idx = header.index("height")
        conf_idx = header.index("conf")
        box_coords = []
        for box in boxes[1:]:
            if len(box) == len(header):
                if float(box[conf_idx]) != -1 and box[text_idx] != "":
                    x, y, w, h = (
                        int(box[left_idx]),
                        int(box[top_idx]),
                        int(box[width_idx]),
                        int(box[height_idx]),
                    )
                    box_coords.append(((x, y), (x + w, y + h)))  # diagonal of the rect
        return box_coords
