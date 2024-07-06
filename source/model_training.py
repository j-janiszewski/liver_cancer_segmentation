from fastai.vision.all import *
from datetime import datetime
import pandas as pd
import os


def foreground_acc(inp, targ, bkg_idx=0, axis=1):  # exclude a background from metric
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()


def cust_foreground_acc(inp, targ):  # # include a background into the metric
    return foreground_acc(
        inp=inp, targ=targ, bkg_idx=3, axis=1
    )  # 3 is a dummy value to include the background which is 0


BS = 16
IM_SIZE = 128

codes = np.array(["background", "liver", "tumor"])


def get_x(fname: Path):
    return fname


def label_func(x):
    return f"./train_masks/{x.stem}_mask.png"


if __name__ == "__main__":

    today = datetime.now()
    model_name = f"{today.day}_{today.month}"
    os.mkdir(f"./models/{model_name}")

    tfms = [IntToFloatTensor(), Normalize()]

    db = DataBlock(
        blocks=(
            ImageBlock(),
            MaskBlock(codes),
        ),  # codes = {"Backround": 0,"Liver": 1,"Tumor": 2}
        batch_tfms=tfms,
        splitter=RandomSplitter(),
        item_tfms=[Resize(IM_SIZE)],
        get_items=get_image_files,
        get_y=label_func,
    )

    dls = db.dataloaders("./train_images", bs=BS, device=default_device(1))

    learn = unet_learner(
        dls,
        resnet34,
        loss_func=CrossEntropyLossFlat(axis=1),  # czy na pewno
        metrics=[foreground_acc, cust_foreground_acc, DiceMulti()],
    )

    lr = learn.lr_find()

    training_metrics_file = f"./models/{model_name}/model_training.csv"

    learn.fine_tune(
        5,
        base_lr=lr.valley,
        wd=0.1,
        cbs=[
            CSVLogger(training_metrics_file),
            SaveModelCallback(),
        ],
    )

    df = pd.read_csv(training_metrics_file)
    df["learning_rate"] = lr.valley
    df.to_csv(training_metrics_file)

    # Save the model
    learn.export(f"./models/{model_name}/Liver_segmentation")
