import os
import cv2
import csv
import pandas as pd
from skimage.feature import hog
from skimage import color, exposure
import random
import matplotlib.pyplot as plt
import pathlib

FACE_IMAGES_DIRECTORY = "UTKFace"
FACES_ZIP_DATA_FILE = os.path.join(pathlib.Path(__file__).parent.absolute(), "HOG_face_data.zip")
FACES_DATA_FILE = os.path.join(pathlib.Path(__file__).parent.absolute(), "HOG_face_data.csv")

if not os.path.isdir(os.path.join(pathlib.Path(__file__).parent.absolute(), FACE_IMAGES_DIRECTORY)):
    print("Error: Could not find ", FACE_IMAGES_DIRECTORY, " directory.")
    print("please download from: https://drive.google.com/file/d/0BxYys69jI14kYVM3aVhKS1VhRUk/view?usp=sharing")
    raise FileNotFoundError


def create_feature_image():
    # reference https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
    sex_lookup = {'0': "male", '1': "female"}
    filename = random.choice(os.listdir(FACE_IMAGES_DIRECTORY))
    print("Generating plot for file ", filename)
    f = os.path.join(FACE_IMAGES_DIRECTORY, filename)
    if os.path.isfile(f) and filename.endswith('.jpg'):
        age, sex, race, _ = filename.split("_")
        bgr = cv2.imread(f)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = color.rgb2gray(rgb)
        fd, hog_image = hog(gray, orientations=6, pixels_per_cell=(33, 33),
                            cells_per_block=(1, 1), visualize=True, multichannel=False, feature_vector=True)
        print(fd.shape)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(rgb, cmap=plt.cm.gray)
        ax1.set_title('Input Image Sex: {}'.format(sex_lookup[sex]))

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.savefig("Figures/HOG_features_visualization.png")
        plt.show()


def generate_HOG_features(write_compressed_file=False):
    orientations = 6
    pixels_per_cell = (33, 33)
    test_file = os.listdir(FACE_IMAGES_DIRECTORY)[0]
    bgr = cv2.imread(os.path.join(FACE_IMAGES_DIRECTORY, test_file))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = color.rgb2gray(rgb)
    fd, hog_image = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=(1, 1), visualize=True, multichannel=False, feature_vector=True)
    NUM_FEATURES = fd.shape[0]

    csvfile = open(FACES_DATA_FILE, 'w')

    faces_data_csv = csv.writer(csvfile)
    columns = ["filename", "age", "sex", "race"]
    for i in range(NUM_FEATURES):
        columns.append("Feature_"+str(i))
    faces_data_csv.writerow(columns)

    count = 0

    for filename in os.listdir(FACE_IMAGES_DIRECTORY):
        print("Encoding File {}".format(count))
        count += 1
        f = os.path.join(FACE_IMAGES_DIRECTORY, filename)
        if os.path.isfile(f) and filename.endswith('.jpg'):
            age, sex, race, _ = filename.split("_")
            row = [filename, age, sex, race]
            bgr = cv2.imread(f)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray = color.rgb2gray(rgb)
            fd, hog_image = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                cells_per_block=(1, 1), visualize=True, multichannel=False, feature_vector=True)
            for i in fd:
                row.append(i)
            faces_data_csv.writerow(row)

    if write_compressed_file:
        df = pd.read_csv(FACES_DATA_FILE)
        df.to_csv(FACES_ZIP_DATA_FILE, compression='gzip')


if __name__ == "__main__":
    # generate_HOG_features(write_compressed_file=True)
    create_feature_image()
