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
from src.utils.bounds import compute_safe_bounds
from src.load_raw import load_raw_to_fieldlat_mesh


STRUCTURE_FAMILIES = {
    "TPMS": {
        "structures": ["gyroid", "diamond", "primitive", "lidinoid"],
        "params": {
            "min_cell_size": (2, 20, 6),
            "max_cell_size": (5, 50, 15),
            "threshold": (0.2, 1.0, 0.4),
            "resolution": (50, 200, 100)
        }
    },

    "Voronoi": {
    "structures": ["voronoi"],
    "params": {
        "seed_count": (50, 1000, 300),
        "final_points": (50, 300, 100)
        }
    },

    "Spinodal": {
        "structures": ["spinodal"],
        "params": {
            "sigma": (1.0, 10.0, 3.0),
            "threshold": (-1.0, 1.0, 0.0),
            "encode_strength": (0.0, 2.0, 0.8),
            "data_smoothing": (0.0, 5.0, 1.0),
            "seed": (0, 9999, 42)
        }
    },

    "Lattice": {
    "structures": ["octet", "bcc", "cubic"],
    "params": {
        "cell_size": (5, 20, 10),
        "thickness": (0.2, 1.0, 0.4)
        }
    }
}


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

        # Family dropdown
        self.family_box = QtWidgets.QComboBox()
        self.family_box.addItems(list(STRUCTURE_FAMILIES.keys()))
        self.family_box.currentTextChanged.connect(self.update_structure_options)
        control_panel.addWidget(self.family_box)

        self.param_layout = QtWidgets.QFormLayout()
        control_panel.addLayout(self.param_layout)

        self.param_widgets = {}

        # Structure dropdown
        self.structure_box = QtWidgets.QComboBox()
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

            # ⭐用 FieldLat loader
            self.fieldlat_mesh = load_raw_to_fieldlat_mesh(file_path)

            # ⭐计算 safe bounds（只针对 TPMS）
            _, safe_min, safe_max = compute_safe_bounds(
                self.fieldlat_mesh,
                resolution=100
            )

            self.safe_min = safe_min
            self.safe_max = safe_max


        # self.plotter.add_mesh(self.mesh, color="lightblue")
        # self.plotter.reset_camera() 
        self.plotter.clear()
        self.plotter.add_mesh(self.mesh)
        self.plotter.reset_camera()

    # -------------------------
    # Generate mesh
    # -------------------------
    def generate(self):
        family = self.family_box.currentText()
        structure = self.structure_box.currentText()
        params = self.get_params()

        # if params["min_cell_size"] >= params["max_cell_size"]:
        #     print("⚠️ Fixing invalid cell size")
        #     params["min_cell_size"] = params["max_cell_size"] * 0.5

        self.mesh = generate_sample(
            raw_path=self.raw_path,
            family=family,
            structure=structure,
            params=params
        )

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

    def update_structure_options(self):
        print("updating structure options")
        family = self.family_box.currentText()

        self.structure_box.clear()
        self.structure_box.addItems(
            STRUCTURE_FAMILIES[family]["structures"]
        )

        self.update_parameter_panel()

    def update_parameter_panel(self):
        print("updating params")
        # clear old
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.param_widgets.clear()
        
        family = self.family_box.currentText()
        params = STRUCTURE_FAMILIES[family]["params"]

        for name, (min_v, max_v, default) in params.items():
            if family == "TPMS" and name == "min_cell_size":
                min_v = self.safe_min
                max_v = self.safe_max

            if family == "TPMS" and name == "max_cell_size":
                min_v = self.safe_min
                max_v = self.safe_max

            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(min_v, max_v)
            spin.setValue(default)
            spin.setSingleStep((max_v - min_v) / 50)

            self.param_layout.addRow(name, spin)
            self.param_widgets[name] = spin

    def get_params(self):
        print("getting params")
        return {
            name: widget.value()
            for name, widget in self.param_widgets.items()
        }


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())