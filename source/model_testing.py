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


def save_confusion_matrix_plot(confusion_matrix, pixel_wise=False):
    ax = sns.heatmap(confusion_matrix, annot=True, cmap="Blues")

    ax.set_title("Confusion Matrix: Liver Tumor detection \n\n")
    ax.set_xlabel("\nActual Values")
    ax.set_ylabel("Predicted Values ")

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(["True", "False"])
    ax.yaxis.set_ticklabels(["True", "False"])
    conf_type = "pixel_wise" if pixel_wise else "classification"
    plt.savefig(f"models/{MODEL_NAME}/confusion_matrix_{conf_type}.png")


TEST_IMAGES_DIR = "test_images"
MODEL_NAME = "7_5"

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
    detected_wrongly = []
    no_tumor_detected_correctly = []
    conf_matrix_classification = np.zeros((2, 2), dtype=int)
    conf_matrix_pixelwise = np.zeros((2, 2), dtype=int)
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
            conf_matrix_classification[0, 0] += 1
            detected_tumors.append(test_file)
        if tumor_p == False and tumor_t == False:
            conf_matrix_classification[1, 1] += 1
            no_tumor_detected_correctly.append(test_file)
        if tumor_p == False and tumor_t == True:
            conf_matrix_classification[1, 0] += 1
            no_detected_tumors.append(test_file)
        if tumor_p == True and tumor_t == False:
            conf_matrix_classification[0, 1] += 1
            detected_wrongly.append(test_file)

        resized_pred = object_codes.repeat(4, axis=0).repeat(4, axis=1)
        is_tumor_p = resized_pred == 2
        is_tumor_t = mask_t == 2

        conf_matrix_pixelwise[0, 0] += np.logical_and(is_tumor_t, is_tumor_p).sum()
        conf_matrix_pixelwise[1, 1] += np.logical_and(
            np.logical_not(is_tumor_t), np.logical_not(is_tumor_p)
        ).sum()
        conf_matrix_pixelwise[0, 1] += np.logical_and(
            np.logical_not(is_tumor_t), is_tumor_p
        ).sum()
        conf_matrix_pixelwise[1, 0] += np.logical_and(
            is_tumor_t, np.logical_not(is_tumor_p)
        ).sum()

    with open(f"models/{MODEL_NAME}/not_detected_tumors.json", "w") as f:
        json.dump(no_detected_tumors, f)

    with open(f"models/{MODEL_NAME}/detected_tumors.json", "w") as f:
        json.dump(detected_tumors, f)

    with open(f"models/{MODEL_NAME}/detected_wrongly.json", "w") as f:
        json.dump(detected_wrongly, f)

    with open(f"models/{MODEL_NAME}/no_tumor_detected_correctly.json", "w") as f:
        json.dump(no_tumor_detected_correctly, f)

    with open(f"models/{MODEL_NAME}/no_tumor_detected_correctly.json", "w") as f:
        json.dump(no_tumor_detected_correctly, f)

    with open(f"models/{MODEL_NAME}/conf_matrix_slice.npy", "wb") as f:
        np.save(f, conf_matrix_classification)

    with open(f"models/{MODEL_NAME}/conf_matrix_pixels.npy", "wb") as f:
        np.save(f, conf_matrix_pixelwise)

    save_confusion_matrix_plot(conf_matrix_classification)
    save_confusion_matrix_plot(conf_matrix_pixelwise, True)
