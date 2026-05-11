import matplotlib
matplotlib.use("TkAgg")

from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
import torch
import numpy as np


def read_mhd(mhd_path):
    meta = {}

    with open(mhd_path, "r") as f:
        for line in f:
            line = line.strip()

            if "=" in line:
                key, value = line.split("=", 1)

                meta[key.strip()] = value.strip()

    dims = list(map(int, meta["DimSize"].split()))

    element_type = meta["ElementType"]

    type_map = {
        "MET_CHAR": np.int8,
        "MET_UCHAR": np.uint8,
        "MET_SHORT": np.int16,
        "MET_USHORT": np.uint16,
        "MET_INT": np.int32,
        "MET_UINT": np.uint32,
        "MET_FLOAT": np.float32,
        "MET_DOUBLE": np.float64,
    }

    dtype = type_map[element_type]

    raw_file = meta["ElementDataFile"]

    raw_path = os.path.join(
        os.path.dirname(mhd_path),
        raw_file
    )

    data = np.fromfile(raw_path, dtype=dtype)

    data = data.reshape(
        (dims[2], dims[1], dims[0])
    )

    return data, meta


class SegmentationDataset(Dataset):
    def __init__(self, data_dir,
                 image_transform=None, mask_transform=None):

        self.data_dir = data_dir

        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.files = []
        self.get_files()

        self.data = []
        self.get_slice_data()

    def __len__(self):
        return len(self.data)

    def get_files(self):
        images = []
        masks = []
        for f in os.listdir(self.data_dir):
            if os.path.isfile(os.path.join(self.data_dir, f)):
                if f.startswith("image") and f.endswith(".mhd"):
                    images.append(f)
                elif f.startswith("labels") and f.endswith(".mhd"):
                    masks.append(f)

        for img in images:
            mask = img.replace("image", "labels")
            if mask in masks:
                self.files.append((img, mask))

    def get_slice_data(self):
        for pair in self.files:
            img_path = os.path.join(self.data_dir, pair[0])
            mask_path = os.path.join(self.data_dir, pair[1])
            img, _ = read_mhd(img_path)
            mask, _ = read_mhd(mask_path)
            if mask.shape == img.shape:
                for idx in range(img.shape[0]):
                    self.data.append((img[idx], mask[idx]))

    def __getitem__(self, idx):

        image = self.data[idx][0]
        mask = self.data[idx][1]

        image = image.astype(np.float32)

        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        mask = mask.astype(np.uint8)
        # TRANSFORMS
        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = mask.squeeze(1).long()

        return image, mask

def load_dataset():

    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((384, 288)),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(
            (384, 288),
            interpolation=transforms.InterpolationMode.NEAREST
        ),
        transforms.PILToTensor()
    ])

    train_dataset = SegmentationDataset(
        data_dir="data_module/dataset/training/training/",
        image_transform=image_transform,
        mask_transform=mask_transform
    )

    validation_dataset = SegmentationDataset(
        data_dir="data_module/dataset/validation/validation/",
        image_transform=image_transform,
        mask_transform=mask_transform
    )

    # SPLIT
    val_size = int(0.5 * len(validation_dataset))
    test_size = len(validation_dataset) - val_size

    val_dataset, test_dataset = random_split(
        validation_dataset,
        [val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=24
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=24
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=24
    )

    # DEBUG ПЕРЕВІРКА
    sample_img, sample_mask = train_dataset[20]

    print("Image shape:", sample_img.shape)
    print("Mask shape:", sample_mask.shape)
    print("Unique mask classes:", torch.unique(sample_mask))

    return train_dataset, train_loader, val_loader, test_loader
