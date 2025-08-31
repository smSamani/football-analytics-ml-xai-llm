from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QPixmap
from pathlib import Path
import sys

def show_k_selection_gui():
    class KSelector(QWidget):
        def __init__(self):
            super().__init__()
            self.selected_k = None
            self.setWindowTitle("Select Optimal K for Clustering")

            layout = QVBoxLayout()
            k_values = [3, 4, 5]
            image_paths = [
                Path("outputs/plots/clustering") / f"KMeans_k{k}" / "cluster_visualization.png"
                for k in k_values
            ]

            row = QHBoxLayout()
            for k, img_path in zip(k_values, image_paths):
                try:
                    pixmap = QPixmap(str(img_path)).scaled(400, 400)
                    label = QLabel()
                    label.setPixmap(pixmap)

                    btn = QPushButton(f"Select K = {k}")
                    btn.clicked.connect(lambda _, val=k: self.select_k(val))

                    col = QVBoxLayout()
                    col.addWidget(label)
                    col.addWidget(btn)
                    row.addLayout(col)
                except Exception as e:
                    print(f"[ERROR] Could not load image for k={k}: {e}")

            layout.addLayout(row)
            self.setLayout(layout)

        def select_k(self, k):
            self.selected_k = k
            self.close()

    app = QApplication(sys.argv)
    selector = KSelector()
    selector.show()
    app.exec()

    return selector.selected_k

if __name__ == "__main__":
    selected_k = show_k_selection_gui()
    print("✅ K Selected by user:", selected_k)