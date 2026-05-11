from ml_module.model_batch_drop import train_model
import sys
import matplotlib
matplotlib.use("TkAgg")

#train_model()

from PyQt6.QtWidgets import QApplication
from UI.ui_module import SegmentationApp


# RUN APPLICATION
app = QApplication(sys.argv)

window = SegmentationApp()

window.show()

sys.exit(app.exec())
