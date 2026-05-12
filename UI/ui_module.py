import sys
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import pydicom
from PIL import Image
from UI.SegmentationEngine import InferenceEngine, read_mhd

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QSlider,
    QProgressDialog,
    QTextEdit,
    QSplitter
)

from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from ml_module.model_batch_drop import UNet
from UI.SegmentationEngine import InferenceEngine, read_mhd


# MAIN WINDOW
class SegmentationApp(QWidget):

    def __init__(self):

        super().__init__()
        self.engine = InferenceEngine()

        self.setWindowTitle("Medical Image Segmentation")
        self.resize(1400, 800)

        self.volume = None
        self.pred_volume = None
        self.current_file_path = None

        # LEFT PANEL (CONTROLS)
        self.load_button = QPushButton("Load File")
        self.segment_button = QPushButton("Segment")
        self.convert_button = QPushButton("Convert to PNG")

        self.load_button.setMinimumHeight(50)
        self.segment_button.setMinimumHeight(50)
        self.convert_button.setMinimumHeight(50)

        self.load_button.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
        """)

        self.segment_button.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
        """)

        self.convert_button.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
        """)

        self.load_button.clicked.connect(self.load_file)
        self.segment_button.clicked.connect(self.segment_volume)
        self.convert_button.clicked.connect(self.convert_current_file)

        self.path_label = QLabel("No file selected")

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.segment_button)
        left_layout.addWidget(self.convert_button)
        left_layout.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # CENTER (IMAGE VIEWER)
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)

        self.slice_label = QLabel("Slice: 0")

        self.slice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.slice_label.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            padding: 10px;
        """)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.update_slice)

        self.slider.setStyleSheet("""
            padding: 10px;
        """)

        image_layout = QVBoxLayout()
        image_layout.addWidget(self.slice_label)
        image_layout.addWidget(self.canvas)
        image_layout.addWidget(self.slider)
        image_layout.addWidget(self.path_label)

        image_widget = QWidget()
        image_widget.setLayout(image_layout)

        # RIGHT PANEL (DICOM INFO)
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Patient Info"))
        right_layout.addWidget(self.info_box)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # SPLITTER (3 PANELS)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(image_widget)
        splitter.addWidget(right_widget)

        splitter.setSizes([250, 800, 350])

        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)

        self.setLayout(main_layout)

    # LOAD FILE
    def load_file(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Medical File",
            "",
            "Medical Files (*.mhd *.dcm)"
        )

        if not file_path:
            return

        self.current_file_path = file_path

        self.path_label.setText(file_path)

        # LOAD MHD
        if file_path.endswith(".mhd"):
            self.volume = read_mhd(file_path)
            self.info_box.clear()

        # LOAD DICOM
        elif file_path.endswith(".dcm"):

            image = self.engine.read_dicom(file_path)
            self.show_dicom_info()

            # 2D
            if len(image.shape) == 2:
                self.volume = np.expand_dims(image, axis=0)

            # 3D
            elif len(image.shape) == 3:
                self.volume = image

            else:
                raise ValueError("Unsupported DICOM shape")

        else:
            QMessageBox.warning(
                self,
                "Error",
                "Unsupported file format"
            )
            self.info_box.clear()
            return

        self.pred_volume = None
        print("Volume shape:", self.volume.shape)

        # Slider setup
        self.slider.setMaximum(
            self.volume.shape[0] - 1
        )

        self.slider.setValue(
            self.volume.shape[0] // 2
        )

        self.update_slice()

    # SEGMENT VOLUME
    def segment_volume(self):

        if self.volume is None:
            QMessageBox.warning(
                self,
                "Error",
                "Load file first"
            )

            return

        total_slices = self.volume.shape[0]
        predictions = []

        # PROGRESS DIALOG
        progress = QProgressDialog(
            "Segmenting...",
            "Cancel",
            0,
            total_slices,
            self
        )

        progress.setWindowTitle(
            "Segmentation Progress"
        )
        progress.setMinimumDuration(0)

        # PROCESS SLICES
        for idx in range(total_slices):

            if progress.wasCanceled():
                return

            image = self.volume[idx]

            prediction = self.engine.predict_slice(image)

            predictions.append(prediction)

            progress.setValue(idx + 1)

            QApplication.processEvents()

        self.pred_volume = np.array(predictions)

        QMessageBox.information(
            self,
            "Done",
            "Segmentation completed"
        )

        self.update_slice()

    def convert_current_file(self):

        if self.current_file_path is None:
            QMessageBox.warning(self, "Error", "Load file first")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")

        if not output_dir:
            return

        try:

            if self.pred_volume is None:
                save_dir = self.engine.convert_to_png(
                    self.current_file_path,
                    output_dir
                )
            else:
                save_dir = self.engine.save_segmented_volume(
                    self.volume,
                    self.pred_volume,
                    self.current_file_path,
                    output_dir
                )

            QMessageBox.information(
                self,
                "Success",
                f"PNG files saved to:\n{save_dir}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Conversion Error",
                str(e)
            )

    # UPDATE DISPLAY
    def update_slice(self):

        if self.volume is None:
            return

        idx = self.slider.value()
        self.slice_label.setText(
            f"Slice: {idx}"
        )

        image = self.volume[idx]

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title("Segmentation Overlay")

        # ORIGINAL IMAGE
        ax.imshow(
            image,
            cmap="gray"
        )

        # OVERLAY MASK
        if self.pred_volume is not None:
            pred = self.pred_volume[idx]

            # resize mask to original image size
            pred_resized = np.array(
                Image.fromarray(
                    pred.astype(np.uint8)
                ).resize(
                    (image.shape[1], image.shape[0]),
                    Image.NEAREST
                )
            )

            # hide background class
            masked_pred = np.ma.masked_where(
                pred_resized == 0,
                pred_resized
            )

            # overlay
            custom_cmap = ListedColormap([
                "black",  # background
                "red",  # upper bone
                "green",  # upper meniscus
                "blue",  # lower meniscus
                "yellow"  # lower bone
            ])

            ax.imshow(
                masked_pred,
                cmap=custom_cmap,
                alpha=0.25,
                vmin=0,
                vmax=4
            )

            legend_elements = [
                Patch(facecolor='red', label='Верхня кістка'),
                Patch(facecolor='green', label='Верхній меніск'),
                Patch(facecolor='blue', label='Нижня кістка'),
                Patch(facecolor='yellow', label='Нижній меніск'),
            ]

            ax.legend(
                handles=legend_elements,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.02),
                fontsize=11,
                frameon=True,
                ncol=4
            )

            ax.axis("off")

        self.canvas.draw()

    def show_dicom_info(self):

        if not hasattr(self.engine, "current_dicom"):
            return

        d = self.engine.current_dicom

        info = []

        tags = [
            "PatientName",
            "PatientID",
            "PatientAge",
            "PatientSex",
            "Modality",
            "StudyDate",
            "BodyPartExamined"
        ]

        for t in tags:
            if hasattr(d, t):
                info.append(f"{t}: {getattr(d, t)}")

        self.info_box.setText("\n".join(info))


def run_app():
    # RUN APPLICATION
    app = QApplication(sys.argv)

    window = SegmentationApp()

    window.show()

    sys.exit(app.exec())

#run_app()
