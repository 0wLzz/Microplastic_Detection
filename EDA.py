import pandas
import cv2
import os

DATASET = "valid"

# Importing
image_name_list = os.listdir(f"archive/{DATASET}/")
image_name_list.remove("_annotations.csv")

df = pandas.read_csv(f"archive/{DATASET}/_annotations.csv")

def load_image(img_name) -> object:
    loaded_img = cv2.imread(f"archive/{DATASET}/" + img_name)
    color = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)

    return color

def label_image(img_name):
    return df[df["filename"] == img_name]


os.makedirs(f"annotated_images/{DATASET}", exist_ok=True)
for img_name in image_name_list: 
    image = load_image(img_name)
    labels = label_image(img_name)

    for index, row in labels.iterrows():
        x_min = row["xmin"]
        y_min = row["ymin"]
        x_max = row["xmax"]
        y_max = row["ymax"]

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"annotated_images/{DATASET}/" + img_name, image_to_save)
