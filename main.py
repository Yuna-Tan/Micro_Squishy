import os
os.environ["QT_API"] = "pyside6"

import sys
import numpy as np
import pyvista as pv

from PySide6 import QtWidgets
from pyvistaqt import QtInteractor

# import your pipeline
from src.generate_sample import generate_sample
from src.utils.mesh_utils import save_mesh
from src.load_raw import load_raw

STRUCTURES = ["gyroid", "diamond", "primitive", "lidinoid", "voronoi", "lattice"]

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Microstructure Generator")
        self.resize(1000, 700)

        # -------- Main layout --------
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QHBoxLayout(central_widget)

        # -------- Left panel (controls) --------
        control_panel = QtWidgets.QVBoxLayout()

        # Load RAW button
        self.load_btn = QtWidgets.QPushButton("Load RAW")
        self.load_btn.clicked.connect(self.load_raw)
        control_panel.addWidget(self.load_btn)

        # Structure selection
        self.structure_box = QtWidgets.QComboBox()
        self.structure_box.addItems(STRUCTURES)
        control_panel.addWidget(self.structure_box)

        # Generate button
        self.generate_btn = QtWidgets.QPushButton("Generate")
        self.generate_btn.clicked.connect(self.generate)
        control_panel.addWidget(self.generate_btn)

        # Export button
        self.export_btn = QtWidgets.QPushButton("Export STL")
        self.export_btn.clicked.connect(self.export_stl)
        control_panel.addWidget(self.export_btn)

        control_panel.addStretch()

        # -------- Right panel (3D viewer) --------
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")

        # -------- Add to layout --------
        layout.addLayout(control_panel, 1)
        layout.addWidget(self.plotter.interactor, 4)

        self.raw_path = None
        self.mesh = None

    # -------------------------
    # Load RAW file
    # -------------------------
    def load_raw(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select RAW file", "", "RAW Files (*.raw)"
        )
        if file_path:
            self.raw_path = file_path
            self.mesh = load_raw(file_path)
            print("Loaded:", file_path)

        # self.plotter.add_mesh(self.mesh, color="lightblue")
        # self.plotter.reset_camera() 
        self.plotter.clear()
        self.plotter.add_mesh(self.mesh)
        self.plotter.reset_camera()

    # -------------------------
    # Generate mesh
    # -------------------------
    def generate(self):
        print("generating")
        if self.raw_path is None:
            print("No RAW loaded")
            return

        family = self.structure_box.currentText()

        self.mesh = generate_sample(
            raw_path=self.raw_path,
            family=family
        )

        if self.mesh is None:
            print("❌ Mesh generation failed")
            return

        # self.plotter.clear()
        # self.plotter.add_mesh(self.mesh)
        # self.plotter.reset_camera()

        self.plotter.clear()

        self.plotter.add_mesh(
            self.mesh,
            color="lightsteelblue",
            show_edges=True,
            edge_color="black",
            smooth_shading=True,
            ambient=0.3,
            diffuse=0.6,
            specular=0.2
        )

        self.plotter.reset_camera()

    # -------------------------
    # Export STL
    # -------------------------
    def export_stl(self):
        if self.mesh is None:
            print("No mesh to export")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save STL", "", "STL Files (*.stl)"
        )

        if file_path:
            save_mesh(self.mesh, file_path)
            print("Saved:", file_path)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())