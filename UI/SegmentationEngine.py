from ml_module.model import UNet
from data_module.DataLoader import read_mhd
import sys
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import pydicom
from PIL import Image
from data_module.DataLoader import read_mhd


class InferenceEngine:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Device:", self.device)

        self.model = UNet().to(self.device)

        self.model.load_state_dict(
            torch.load(
                "best_unet_model_5.pth",
                map_location=self.device
            )
        )

        self.model.eval()

        print("Model loaded")

        self.current_dicom = None

    def read_dicom(self, dicom_path):
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array.astype(np.float32)

        # save metadata for UI
        self.current_dicom = dicom

        return image

    def read_mhd(self, mhd_path):
        data, _ = read_mhd(mhd_path)
        return data

    def predict_slice(self, image):

        input_tensor = self.normalize_image(image)

        with torch.no_grad():
            output = self.model(input_tensor)

            prediction = torch.argmax(output, dim=1)

        prediction = (
            prediction
            .squeeze()
            .cpu()
            .numpy()
        )

        return prediction

    def normalize_image(self, image):
        image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 288)),
            transforms.ToTensor()
        ])

        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # TRANSFORM
        input_tensor = image_transform(image)
        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .to(self.device)
        )

        return input_tensor

    def save_segmented_volume(
            self,
            volume,
            pred_volume,
            input_path,
            output_dir
    ):

        filename = os.path.splitext(os.path.basename(input_path))[0]

        save_dir = os.path.join(
            output_dir,
            f"{filename}_segmented"
        )

        os.makedirs(save_dir, exist_ok=True)

        colors = {
            1: [255, 0, 0],  # upper bone
            2: [0, 255, 0],  # upper meniscus
            3: [0, 0, 255],  # lower meniscus
            4: [255, 255, 0],  # lower bone
        }

        total_slices = volume.shape[0]

        for idx in range(total_slices):

            image = volume[idx]
            pred = pred_volume[idx]

            # resize mask
            pred = np.array(
                Image.fromarray(pred.astype(np.uint8)).resize(
                    (image.shape[1], image.shape[0]),
                    Image.NEAREST
                )
            )

            # normalize image
            image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
            image_norm = (image_norm * 255).astype(np.uint8)

            rgb = np.stack([image_norm] * 3, axis=-1)
            overlay = rgb.copy()

            # apply colors
            for cls, color in colors.items():
                mask = pred == cls
                overlay[mask] = (
                        0.7 * overlay[mask] + 0.3 * np.array(color)
                )

            save_path = os.path.join(save_dir, f"slice_{idx:03d}.png")

            Image.fromarray(overlay.astype(np.uint8)).save(save_path)

        return save_dir

    def convert_to_png(self, input_path, output_dir):

        # LOAD IMAGE
        if input_path.endswith(".dcm"):

            dicom = pydicom.dcmread(input_path)
            volume = dicom.pixel_array.astype(np.float32)

            patient_name = str(getattr(dicom, "PatientName", "unknown"))
            study_date = str(getattr(dicom, "StudyDate", "unknown_date"))

            filename = os.path.splitext(os.path.basename(input_path))[0]

        elif input_path.endswith(".mhd"):

            volume = read_mhd(input_path).astype(np.float32)

            patient_name = "mhd_patient"
            study_date = "unknown_date"

            filename = os.path.splitext(os.path.basename(input_path))[0]

        else:
            raise ValueError("Unsupported format")

        # NORMALIZATION
        volume = (
                         volume - volume.min()
                 ) / (
                         volume.max() - volume.min() + 1e-8
                 )

        volume = (volume * 255).astype(np.uint8)

        # OUTPUT FOLDERS
        info = f"{patient_name}_{study_date}"

        patient_dir = os.path.join(
            output_dir,
            info
        )

        os.makedirs(patient_dir, exist_ok=True)

        scan_dir = os.path.join(
            patient_dir,
            filename
        )

        os.makedirs(scan_dir, exist_ok=True)

        # SAVE PNG
        # 3D volume
        if volume.ndim == 3:

            num_slices = volume.shape[0]

            for i in range(num_slices):
                slice_img = volume[i]

                slice_path = os.path.join(
                    scan_dir,
                    f"slice_{i:03d}.png"
                )

                Image.fromarray(slice_img).save(slice_path)

            print(f"Saved {num_slices} slices")

        # 2D image
        elif volume.ndim == 2:

            slice_path = os.path.join(
                scan_dir,
                "slice_000.png"
            )

            Image.fromarray(volume).save(slice_path)
            print("Saved 2D image")

        else:
            raise ValueError(
                f"Unsupported shape: {volume.shape}"
            )

        return scan_dir
