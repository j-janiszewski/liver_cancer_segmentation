from fastai.vision.all import *
import os
import seaborn as sns
from tqdm import tqdm
import json
from model_training import (
    label_func,
    foreground_acc,
    cust_foreground_acc,
    IM_SIZE,
)


def save_confusion_matrix_plot(confusion_matrix):
    ax = sns.heatmap(confusion_matrix, annot=True, cmap="Blues")

    ax.set_title("Confusion Matrix: Liver Tumor detection \n\n")
    ax.set_xlabel("\nActual Values")
    ax.set_ylabel("Predicted Values ")

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(["True", "False"])
    ax.yaxis.set_ticklabels(["True", "False"])
    plt.savefig(f"models/{MODEL_NAME}/confusion_matrix.png")


TEST_IMAGES_DIR = "test_images"
MODEL_NAME = "24_4"

if __name__ == "__main__":
    # loading the tensor flow model
    tfms = [Resize(IM_SIZE), IntToFloatTensor(), Normalize()]
    learn0 = load_learner(f"./models/{MODEL_NAME}/Liver_segmentation", cpu=False)

    learn0.dls.transform = tfms
    test_files = os.listdir(TEST_IMAGES_DIR)
    test_dl = learn0.dls.test_dl([f"{TEST_IMAGES_DIR}/" + file for file in test_files])
    preds, y = learn0.get_preds(dl=test_dl)

    no_detected_tumors = []
    detected_tumors = []
    conf_matrix = np.zeros((2, 2), dtype=int)
    ids = [int(file.split("_")[0].split("-")[1]) for file in test_files]
    print("calculating stats regarding predictions...")

    for test_file, pred in tqdm(zip(test_files, preds)):

        mask_t = Image.open(f"test_masks/{test_file[:-4]}_mask.png")
        mask_t = np.array(mask_t)
        unique_objects_t = np.unique(mask_t)

        predicted_mask = np.argmax(pred, axis=0)
        object_codes = np.array(predicted_mask)
        unique_objects_p = np.unique(object_codes)

        tumor_p = True if 2 in unique_objects_p else False
        tumor_t = True if 2 in unique_objects_t else False

        if tumor_p == True and tumor_t == True:
            conf_matrix[0, 0] += 1
            detected_tumors.append(test_file)
        if tumor_p == False and tumor_t == False:
            conf_matrix[1, 1] += 1
        if tumor_p == False and tumor_t == True:
            conf_matrix[1, 0] += 1
            no_detected_tumors.append(test_file)
        if tumor_p == True and tumor_t == False:
            conf_matrix[0, 1] += 1

    with open(f"models/{MODEL_NAME}/not_detected_tumors.json", "w") as f:
        json.dump(no_detected_tumors, f)

    with open(f"models/{MODEL_NAME}/detected_tumors.json", "w") as f:
        json.dump(detected_tumors, f)

    save_confusion_matrix_plot(conf_matrix)
