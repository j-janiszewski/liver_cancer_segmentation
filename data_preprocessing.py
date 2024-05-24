import os
import pandas as pd
import nibabel as nib
from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *
from tqdm import tqdm

NUM_OF_TRAIN_IMGS = 91

# Preprocessing functions
# Source https://docs.fast.ai/medical.imaging


class TensorCTScan(TensorImageBW):
    _show_args = {"cmap": "bone"}


@patch
def freqhist_bins(self: Tensor, n_bins=100):
    "A function to split the range of pixel values into groups, such that each group has around the same number of pixels"
    imsd = self.view(-1).sort()[0]
    t = torch.cat(
        [
            tensor([0.001]),
            torch.arange(n_bins).float() / n_bins + (1 / 2 / n_bins),
            tensor([0.999]),
        ]
    )
    t = (len(imsd) * t).long()
    return imsd[t].unique()


@patch
def hist_scaled(self: Tensor, brks=None):
    "Scales a tensor using `freqhist_bins` to values between 0 and 1"
    if self.device.type == "cuda":
        return self.hist_scaled_pt(brks)
    if brks is None:
        brks = self.freqhist_bins()
    ys = np.linspace(0.0, 1.0, len(brks))
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0.0, 1.0)


@patch
def to_nchan(x: Tensor, wins, bins=None):
    res = [x.windowed(*win) for win in wins]
    if not isinstance(bins, int) or bins != 0:
        res.append(x.hist_scaled(bins).clamp(0, 1))
    dim = [0, 1][x.dim() == 3]
    return TensorCTScan(torch.stack(res, dim=dim))  # ogarnij


@patch
def save_jpg(x: Tensor, path, wins, bins=None, quality=90):
    fn = Path(path).with_suffix(".jpg")
    x = (x.to_nchan(wins, bins) * 255).byte()
    im = Image.fromarray(
        x.permute(1, 2, 0).numpy(), mode=["RGB", "CMYK"][x.shape[0] == 4]
    )
    im.save(fn, quality=quality)


# Preprocess the nii file
# Source https://docs.fast.ai/medical.imaging

dicom_windows = types.SimpleNamespace(
    brain=(80, 40),
    subdural=(254, 100),
    stroke=(8, 32),
    brain_bone=(2800, 600),
    brain_soft=(375, 40),
    lungs=(1500, -600),
    mediastinum=(350, 50),
    abdomen_soft=(400, 50),
    liver=(150, 30),
    spine_soft=(250, 50),
    spine_bone=(1800, 400),
    custom=(200, 60),
)


@patch
def windowed(self: Tensor, w, l):
    px = self.clone()
    px_min = l - w // 2
    px_max = l + w // 2
    px[px < px_min] = px_min
    px[px > px_max] = px_max
    return (px - px_min) / (px_max - px_min)


def read_nii(filepath):
    """
    Reads .nii file and returns pixel array
    """
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array


def create_images_and_masks(df_with_files, dir_names="train"):
    slice_sum = 0
    os.makedirs(f"{dir_names}_images", exist_ok=True)
    os.makedirs(f"{dir_names}_masks", exist_ok=True)
    for ii in tqdm(range(0, len(df_with_files))):
        curr_ct = read_nii(
            df_with_files.loc[ii, "dirname"] + "/" + df_with_files.loc[ii, "filename"]
        )
        curr_mask = read_nii(
            df_with_files.loc[ii, "mask_dirname"]
            + "/"
            + df_with_files.loc[ii, "mask_filename"]
        )
        curr_file_name = str(df_with_files.loc[ii, "filename"]).split(".")[0]
        curr_dim = curr_ct.shape[2]  # 512, 512, curr_dim
        slice_sum = slice_sum + curr_dim

        for curr_slice in range(0, curr_dim, 2):  # export every 2nd slice
            data = tensor(curr_ct[..., curr_slice].astype(np.float32))
            mask = Image.fromarray(curr_mask[..., curr_slice].astype("uint8"), mode="L")
            data.save_jpg(
                f"{dir_names}_images/{curr_file_name}_slice_{curr_slice}.jpg",
                [dicom_windows.liver, dicom_windows.custom],
            )
            mask.save(f"{dir_names}_masks/{curr_file_name}_slice_{curr_slice}_mask.png")
    print(slice_sum)


if __name__ == "__main__":

    # Create a meta file for nii files processing

    file_list = []
    for dirname, _, filenames in os.walk("../data/"):
        for filename in filenames:
            if filename != ".DS_Store":
                file_list.append((dirname, filename))

    df_files = pd.DataFrame(file_list, columns=["dirname", "filename"])

    df_files["mask_dirname"] = ""
    df_files["mask_filename"] = ""

    for i in range(131):
        ct = f"volume-{i}.nii"
        mask = f"segmentation-{i}.nii"

        df_files.loc[df_files["filename"] == ct, "mask_filename"] = mask
        df_files.loc[df_files["filename"] == ct, "mask_dirname"] = (
            "../data/segmentations"
        )

    df_files_test = df_files[df_files.mask_filename == ""]
    # drop segment rows
    df_files = (
        df_files[df_files.mask_filename != ""]
        .sort_values(by=["filename"])
        .reset_index(drop=True)
    )
    df_files = df_files.sample(frac=1)

    train_files = df_files[:NUM_OF_TRAIN_IMGS]
    test_files = df_files[NUM_OF_TRAIN_IMGS:]
    train_files = train_files.reset_index(drop=True)
    test_files = test_files.reset_index(drop=True)
    create_images_and_masks(train_files, dir_names="train")
    create_images_and_masks(test_files, dir_names="test")
