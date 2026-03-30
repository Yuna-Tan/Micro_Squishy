# import os
# os.environ["QT_API"] = "pyside6"
# os.environ["QT_MAC_WANTS_LAYER"] = "1" 

from matplotlib import pyplot as plt
import pyvista
import numpy
import matplotlib.pyplot as plt
import squishicalization.voronoi_gpu as voronoi_gpu
from scipy.signal import argrelextrema
from datetime import datetime
import math 
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor
import sys

class Gui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.data = pyvista.ImageData()

        # Frame for rendering        
        self.frame = QtWidgets.QFrame()
        self.top_layout = QtWidgets.QVBoxLayout(self)
        self.top_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.top_widget)
        self.plotter = QtInteractor(self.frame)

        # Tab Panel for setting
        self.tabs = QtWidgets.QStackedWidget()

        # Pane for read data
        self.widget_read = QtWidgets.QWidget()
        self.layout_read = QtWidgets.QFormLayout(self.widget_read)
        self.label_x_scale = QtWidgets.QLabel("X scale")
        self.label_y_scale = QtWidgets.QLabel("Y scale")
        self.label_z_scale = QtWidgets.QLabel("Z scale")
        self.text_x_scale = QtWidgets.QLineEdit("1")
        self.text_y_scale = QtWidgets.QLineEdit("1")
        self.text_z_scale = QtWidgets.QLineEdit("1")
        self.button_load = QtWidgets.QPushButton("Load")
        self.button_load.clicked.connect(self.load_data)
        self.button_load_toy = QtWidgets.QPushButton("Load Toy")
        self.button_load_toy.clicked.connect(self.load_toy_data)

        self.layout_read.addRow(self.label_x_scale,self.text_x_scale)
        self.layout_read.addRow(self.label_y_scale,self.text_y_scale)
        self.layout_read.addRow(self.label_z_scale,self.text_z_scale)
        self.layout_read.addWidget(self.button_load)
        self.layout_read.addWidget(self.button_load_toy)

        # Pane for stencil creation
        self.widget_stencil = QtWidgets.QWidget()
        self.layout_stencil = QtWidgets.QFormLayout(self.widget_stencil)
        self.text_stencil_threshold = QtWidgets.QLineEdit("1")
        self.button_stencil_thresh = QtWidgets.QPushButton("Treshold")
        self.button_stencil_thresh.clicked.connect(self.thresh_stencil)
        self.button_stencil_erode = QtWidgets.QPushButton("Erode")
        self.button_stencil_erode.clicked.connect(self.erode_stencil)
        self.button_stencil_dilate = QtWidgets.QPushButton("Dilate")
        self.button_stencil_dilate.clicked.connect(self.dilate_stencil)

        self.layout_stencil.addRow(QtWidgets.QLabel("Threshold"),self.text_stencil_threshold)
        # self.layout_stencil.addWidget(self.button_load)
        # self.layout_stencil.addRow(self.label_y_scale,self.text_y_scale)
        # self.layout_stencil.addRow(self.label_z_scale,self.text_z_scale)
        self.layout_stencil.addWidget(self.button_stencil_thresh)
        self.layout_stencil.addWidget(self.button_stencil_erode)
        self.layout_stencil.addWidget(self.button_stencil_dilate)
        
        # Pane for threshold data
        self.widget_thresh = QtWidgets.QWidget()
        self.layout_thresh = QtWidgets.QFormLayout(self.widget_thresh)
        self.label_thresh = QtWidgets.QLabel("Threshold")
        self.label_slider_thresh_value = QtWidgets.QLabel("0.1")
        self.slider_thresh = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)        
        self.slider_thresh.setMinimum(0)
        self.slider_thresh.setMaximum(100)
        self.slider_thresh.setValue(1)
        self.slider_thresh.setSingleStep(1)
        self.slider_thresh.setPageStep(10)
        self.slider_thresh.valueChanged.connect(self.thresh_value_changed)
        self.label_iso = QtWidgets.QLabel("Iso")
        self.label_slider_iso_value = QtWidgets.QLabel("0.1")
        self.slider_iso = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)        
        self.slider_iso.setMinimum(0)
        self.slider_iso.setMaximum(100)
        self.slider_iso.setValue(1)
        self.slider_iso.setSingleStep(1)
        self.slider_iso.setPageStep(10)
        self.slider_iso.valueChanged.connect(self.iso_value_changed)        
        self.label_x_lim = QtWidgets.QLabel("X limit")
        self.text_x_lim = QtWidgets.QLineEdit("100")
        self.label_y_lim = QtWidgets.QLabel("Y limit")
        self.text_y_lim = QtWidgets.QLineEdit("100")
        self.label_z_lim = QtWidgets.QLabel("Z limit")
        self.text_z_lim = QtWidgets.QLineEdit("100")
        self.button_thresh = QtWidgets.QPushButton("Threshold")
        self.button_thresh.clicked.connect(self.thresh_data)
        self.button_extract_iso = QtWidgets.QPushButton("Extract Iso Surface")
        self.button_extract_iso.clicked.connect(self.extract_iso)

        self.layout_thresh.addWidget(self.label_slider_thresh_value)
        self.layout_thresh.addRow(self.label_thresh,self.slider_thresh)
        self.layout_thresh.addWidget(self.label_slider_iso_value)
        self.layout_thresh.addRow(self.label_iso,self.slider_iso)
        self.layout_thresh.addRow(self.label_x_lim,self.text_x_lim)
        self.layout_thresh.addRow(self.label_y_lim,self.text_y_lim)
        self.layout_thresh.addRow(self.label_z_lim,self.text_z_lim)
        self.layout_thresh.addWidget(self.button_thresh)
        self.layout_thresh.addWidget(self.button_extract_iso)

        #Pane for sampling
        self.widget_sample = QtWidgets.QWidget()
        self.layout_sample = QtWidgets.QGridLayout(self.widget_sample)
        self.label_num_seed_points = QtWidgets.QLabel("Seed Size")
        self.text_num_seed_points = QtWidgets.QLineEdit("300")
        self.label_num_points = QtWidgets.QLabel("Sample Size")
        self.text_num_points = QtWidgets.QLineEdit("100")
        self.list_tf_points = QtWidgets.QListWidget()
        # self.list_tf_points.setSortingEnabled(True)
        self.list_tf_points.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.text_tf_x = QtWidgets.QLineEdit("0")
        self.text_tf_y = QtWidgets.QLineEdit("0")
        self.label_tf_xy = QtWidgets.QLabel("x,y")
        # self.list_tf_points.addItem("0.0,30.0")
        # self.list_tf_points.addItem("0.3,5.0")
        # self.list_tf_points.addItem("1.0,5.0")
        self.button_add_tf_point = QtWidgets.QPushButton("Add Point")
        self.button_add_tf_point.clicked.connect(self.addTfPoint)
        self.button_del_tf_point = QtWidgets.QPushButton("Delete Point")
        self.button_del_tf_point.clicked.connect(self.delTfPoint)
        self.check_samplethresh = QtWidgets.QCheckBox("Use threshold minimum for sampling")
        self.button_sample = QtWidgets.QPushButton("Sample")
        self.button_sample.clicked.connect(self.samplePoints)

        self.layout_sample.addWidget(self.list_tf_points,0,0,1,2)
        self.layout_sample.setRowStretch(0,-1)
        self.layout_sample.addWidget(self.text_tf_x,1,0)
        self.layout_sample.addWidget(self.text_tf_y,1,1)
        self.layout_sample.addWidget(QtWidgets.QLabel("x"),2,0)
        self.layout_sample.addWidget(QtWidgets.QLabel("y"),2,1)
        self.layout_sample.addWidget(self.button_add_tf_point,3,0)
        self.layout_sample.addWidget(self.button_del_tf_point,3,1)
        self.layout_sample.addWidget(self.label_num_seed_points,4,0)
        self.layout_sample.addWidget(self.text_num_seed_points,4,1)
        self.layout_sample.addWidget(self.label_num_points,5,0)
        self.layout_sample.addWidget(self.text_num_points,5,1)
        self.layout_sample.addWidget(self.check_samplethresh,6,0,1,2)
        self.layout_sample.addWidget(self.button_sample,7,0,1,2)

        #Pane for tesselation

        self.widget_tesselate = QtWidgets.QWidget()
        self.layout_tesselate = QtWidgets.QFormLayout(self.widget_tesselate)
        self.button_tesselate = QtWidgets.QPushButton("Tesselate")
        self.button_tesselate.clicked.connect(self.tesselateData)

        self.layout_tesselate.addWidget(self.button_tesselate)

        #Pane for meshing

        self.widget_mesh = QtWidgets.QWidget()
        self.layout_mesh = QtWidgets.QFormLayout(self.widget_mesh)
        self.button_mesh = QtWidgets.QPushButton("Mesh")
        self.button_save = QtWidgets.QPushButton("Save")
        self.button_mesh.clicked.connect(self.createMesh)
        self.button_save.clicked.connect(self.saveMesh)

        self.layout_mesh.addWidget(self.button_mesh)
        self.layout_mesh.addWidget(self.button_save)

        #Pane for Buttons
        self.widget_navpane = QtWidgets.QWidget()
        self.layout_navpane = QtWidgets.QHBoxLayout(self.widget_navpane)
        self.button_nav_load = QtWidgets.QPushButton("Loading")
        self.button_nav_load.clicked.connect(self.nav_load)
        self.button_nav_thresh = QtWidgets.QPushButton("Filtering")
        self.button_nav_thresh.clicked.connect(self.nav_thresh)
        self.button_nav_sample = QtWidgets.QPushButton("Sampling")
        self.button_nav_sample.clicked.connect(self.nav_sample)
        self.button_nav_tesselate = QtWidgets.QPushButton("Tesselation")
        self.button_nav_tesselate.clicked.connect(self.nav_tesselate)
        self.button_nav_mesh = QtWidgets.QPushButton("Meshing")
        self.button_nav_mesh.clicked.connect(self.nav_mesh)

        self.layout_navpane.addWidget(self.button_nav_load)
        self.layout_navpane.addWidget(self.button_nav_thresh)
        self.layout_navpane.addWidget(self.button_nav_sample)
        self.layout_navpane.addWidget(self.button_nav_tesselate)
        self.layout_navpane.addWidget(self.button_nav_mesh)
        # self.button = QtWidgets.QPushButton("Test")
        # self.slider_label = QtWidgets.QLabel()

        # self.slider = QtWidgets.QSlider()        
        # self.slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        # self.slider.setMinimum(1)
        # self.slider.setMaximum(1000)
        # self.slider.setValue(50)
        
        # self.slider.valueChanged.connect(self.update_slider)
        # self.layout_read.addWidget(self.slider)
        # self.layout_read.addWidget(self.slider_label)
        # self.layout_read.addWidget(self.button)

        self.tabs.addWidget(self.widget_read)
        self.tabs.addWidget(self.widget_thresh)
        self.tabs.addWidget(self.widget_sample)
        self.tabs.addWidget(self.widget_tesselate)
        self.tabs.addWidget(self.widget_mesh)

        
        self.top_layout.addWidget(self.top_widget)
        self.layout.addWidget(self.plotter.interactor, stretch=1)
        self.layout.addWidget(self.tabs)
        self.top_layout.addWidget(self.widget_navpane)
                
        self.setLayout(self.top_layout)

        # self.button_stencil_thresh.clicked.connect(self.thresh_stencil)
        # self.button_stencil_erode = QtWidgets.QPushButton("Erode")
        # self.button_stencil_erode.clicked.connect(self.erode_stencil)
        # self.button_stencil_dilate = QtWidgets.QLineEdit("Dilate")
        # self.button_stencil_dilate.clicked.connect(self.dilate_stencil)

        
    @QtCore.Slot()
    def thresh_stencil(self):
        pass

    @QtCore.Slot()
    def erode_stencil(self):
        pass

    @QtCore.Slot()
    def dilate_stencil(self):
        pass

    @QtCore.Slot()
    def addTfPoint(self):
        point_x = float(self.text_tf_x.text())
        point_y = float(self.text_tf_y.text())
        self.list_tf_points.addItem(str(point_x) + ","+str(point_y))
        # self.list_tf_points.sortItems()

    @QtCore.Slot()
    def delTfPoint(self):        
        self.list_tf_points.takeItem(self.list_tf_points.currentRow())

    @QtCore.Slot()
    def nav_load(self):
        self.tabs.setCurrentWidget(self.widget_read)
    
    @QtCore.Slot()    
    def nav_thresh(self):
        self.tabs.setCurrentWidget(self.widget_thresh)
        
    @QtCore.Slot()
    def nav_sample(self):
        self.tabs.setCurrentWidget(self.widget_sample)

    @QtCore.Slot()
    def nav_mesh(self):
        self.tabs.setCurrentWidget(self.widget_mesh)

    @QtCore.Slot()
    def nav_tesselate(self):
        self.tabs.setCurrentWidget(self.widget_tesselate)

    @QtCore.Slot()
    def update_slider(self):
        
       self.slider_label.setText(str(self.slider.value()))

    def thresh_value_changed(self):
        
       self.label_slider_thresh_value.setText(str(self.slider_thresh.value()))
    def iso_value_changed(self):
        
       self.label_slider_iso_value.setText(str(self.slider_iso.value()))

    @QtCore.Slot()
    def load_toy_data(self):

        # get_sphere_cube()

        self.spacing = [float(self.text_x_scale.text()),float(self.text_y_scale.text()),float(self.text_z_scale.text())]
        
        # test = numpy.ones((300,300,300)) * math.e
        # test  = test  ** (numpy.arange(0,300,1)/300)
  

        x = numpy.linspace(-100, 100, 100)
        y = numpy.linspace(-100, 100, 100)
        z = numpy.linspace(-100, 100, 100)

        # x = numpy.linspace(1, 2, 100)
        # y = numpy.linspace(1, 2, 100)
        # z = numpy.linspace(1, 2, 100)

        # x = numpy.linspace(0, 1, 100)
        # y = numpy.linspace(0, 1, 100)
        # z = numpy.linspace(0, 1, 100)

        xx,yy,zz = numpy.meshgrid(x,y,z)
        #distance
        test = (xx - 0.5) ** 2 + (yy- 0.5) ** 2 + (zz- 0.5) **2

        #quadric
        a = numpy.repeat(1,20)
        # a = numpy.random.rand(9)
        print( a )

        # test = a[0] * xx + a[1] * yy + a[2] * zz + a[3] * xx * yy +  a[4] * yy * zz + a[5] * xx * zz + a[6] * xx**2 + a[7] * yy **2 + a[8] * zz **2

        
        test = xx ** 2 + yy ** 2 - zz**2

        # test =  (yy)
        test =  (zz + xx + yy)
        # test =  (zz)

        # test =  numpy.ones_like(zz)

        # test[xx + zz > 3] = 2
        # test[(xx + yy + zz) > 3] = 2
        # test[(-0.5*xx - 0.5*yy + zz) > 0] = 2

        res = (100,100,100)

        diam = math.floor(min(res)/3)

        cube = numpy.ones(res)*0.1

        n_spheres = 0

        actors = []
        actors = []
        positions = [[50,50,50],[150,150,150]]


        for i in range(0,n_spheres):
            sphere = numpy.zeros((diam,diam,diam))

            x = numpy.linspace(-1, 1, diam)
            y = numpy.linspace(-1, 1, diam)
            z = numpy.linspace(-1, 1, diam)
            
            xx,yy,zz = numpy.meshgrid(x,y,z)

            sphere[xx ** 2 + yy ** 2 + zz**2 <=1] = i +1

            # pos = numpy.random.random_integers(0,res[0]-diam,(3)) 
            pos  = numpy.array(positions[i]).T
            print(pos)
            # cube[225:275,225:275,225:275] = sphere
            cube[pos[0]:pos[0]+diam,pos[1]:pos[1]+diam,pos[2]:pos[2]+diam] += sphere
            center = (pos[0]+diam/2, pos[1]+diam/2, pos[2]+diam/2)
            actors.append(pyvista.Sphere(radius=diam/2, center=center))



        # print(zz.shape)

        # test = (test-test.min())/(test.max() - test.min())
        
        # test = numpy.ones((100,100,100))
        # print(test.shape)
        # test[:,:,0:50] = 2

        # self.data = numpy.pad(test,pad_width=5,constant_values=0)
        self.data = test
        # self.data = cube

        stencil_thresh = self.data.min()

        self.stencil = numpy.zeros_like(self.data)
        self.stencil[self.data >= stencil_thresh] = 1
        self.stencil = cleanVolume(self.stencil)


        # stencil_mesh = pyvista.wrap(self.stencil).contour()
        # stencil_mesh = stencil_mesh.connectivity('largest')
        # stencil_mesh.plot()

        # self.data[self.stencil == 0] = 0        


        self.threshed_data = numpy.copy(self.data)

        self.volume = pyvista.ImageData(pyvista.wrap(self.data))
        self.volume.spacing = self.spacing


        self.plotter.clear()
        # for mesh in actors:
        #     self.plotter.add_mesh(mesh, opacity = 1)
        self.plotter.add_volume(self.volume, opacity = [0.1,0.1])    

        stencil_mesh = pyvista.Cube(center=self.volume.center, x_length=res[0], y_length=res[1], z_length=res[2])
        actors.insert(0,stencil_mesh)

        solution_mesh = pyvista.merge(actors)
        

        pass

    @QtCore.Slot()
    def load_data(self):
        self.spacing = [float(self.text_x_scale.text()),float(self.text_y_scale.text()),float(self.text_z_scale.text())]
        path = QtWidgets.QFileDialog.getOpenFileName(self, caption="Select file", filter="*.raw")[0]
        self.data = load_raw(path)

        # self.data = cleanVolume(self.data)

        stencil_thresh = 0.00

        self.stencil = numpy.zeros_like(self.data)
        self.stencil[self.data > stencil_thresh] = 1
        # self.stencil = numpy.pad(cleanVolume(self.stencil), 1)

        stencil_mesh = pyvista.wrap(self.stencil).contour_labeled(smoothing = True)
        stencil_mesh = stencil_mesh.connectivity('largest')
        # stencil_mesh = stencil_mesh.fill_holes(255)

        # stencil_mesh.save("stencil1.stl")

        # self.data[self.stencil == 0] = 0

        self.slider_thresh.setMinimum(self.data.min())
        self.slider_thresh.setMaximum(self.data.max())
        self.slider_iso.setMinimum(self.data.min())
        self.slider_iso.setMaximum(self.data.max())
        self.list_tf_points.addItem(str(self.data.min()) +",10.0")
        self.list_tf_points.addItem(str(self.data.max()) +",1.0")
        self.text_x_lim.setText(str(self.data.shape[0]))
        self.text_y_lim.setText(str(self.data.shape[1]))
        self.text_z_lim.setText(str(self.data.shape[2]))

        self.threshed_data = numpy.copy(self.data)


        #lut for teaser

        values = [2,300,330,1200,1300,2000,4095]
        
        opacities = [0,0,.1,.2,.5,0,0]

        lut = pyvista.LookupTable(cmap='inferno')
        lut.values = values
        lut.value_range = (0,1)
        lut.hue_range = (0.5,0.6)
        lut.saturation_range = (1,1)
        lut.alpha_range = (0.7,0.7)
        lut.scalar_range = (0,4096)
        lut.apply_opacity(opacities)
        # lut.apply_cmap(['blue','blue','blue','blue','blue','blue','blue'])
        # lut.alpha_range = tuple(opacities)


        self.volume = pyvista.ImageData(pyvista.wrap(self.data))
        self.volume.spacing = self.spacing
        self.plotter.clear()
        # self.plotter.add_volume(self.volume, cmap = lut)  
        self.plotter.add_volume(self.volume)    

    @QtCore.Slot()
    def extract_iso(self):

        thresh = self.slider_iso.value()
        iso_mesh = self.volume.contour([thresh], method='flying_edges')
        self.plotter.clear()

        self.plotter.add_mesh(iso_mesh)

        iso_mesh.save("iso.stl")
    @QtCore.Slot()
    def thresh_data(self):
        thresh = self.slider_thresh.value()
        x_lim = int(self.text_x_lim.text())
        y_lim = int(self.text_y_lim.text())
        z_lim = int(self.text_z_lim.text())
        

        print(thresh)
        print("start thresh")        
        self.data = numpy.pad(self.data, 1)
        self.threshed_data = numpy.copy(self.data)
        
        self.threshed_data[self.data < thresh] = 0

        if(x_lim > 0):self.threshed_data = self.threshed_data[0:x_lim,:,:]
        if(y_lim > 0):self.threshed_data = self.threshed_data[:,0:y_lim,:]
        if(z_lim > 0):self.threshed_data = self.threshed_data[:,:,0:z_lim]

        # self.threshed_data[self.threshed_data > 0] = 1
        print("end thresh")        
        threshed_volume = pyvista.ImageData(pyvista.wrap(self.threshed_data))
        # threshed_volume = threshed_volume.image_dilate_erode(0,1)
        # threshed_volume = threshed_volume.image_dilate_erode(0,1)
        # # threshed_volume = threshed_volume.image_dilate_erode(1,0)
        # threshed_volume = threshed_volume.image_dilate_erode(1,0)
        threshed_volume.spacing = self.spacing
        self.plotter.clear()

        self.plotter.add_volume(threshed_volume)
        # threshed_mesh = threshed_volume.contour([thresh], method='flying_edges')
        # threshed_mesh = threshed_volume.connectivity("largest")
        # self.plotter.add_mesh(threshed_mesh)

    @QtCore.Slot()
    def samplePoints(self):      
        self.data = self.threshed_data
        params = [[],[]]

        for elm in range(0,self.list_tf_points.count()):
            text = self.list_tf_points.item(elm).text()
            # print(self.list_tf_points.item(elm).text())
            point = text.split(",")

            p0 = (float(point[0])-self.data.min())/(self.data.max() - self.data.min())
            params[0].append(p0)
            params[1].append(float(point[1]))

        print(params)

        # min_dist = float(self.text_min_dist.text())
        # max_dist = float(self.text_max_dist.text())
        n_seed_points = int(self.text_num_seed_points.text())
        self.n_points = int(self.text_num_points.text())
            
        # limit data to minimum scalar value to be eligible for seed
        seedlim = self.data.min()
        if(self.check_samplethresh.isChecked()):
            seedlim = self.slider_thresh.value()
            
            self.stencil = numpy.zeros_like(self.data)
            self.stencil[self.data > seedlim] = 1
        # print(seedlim)

        limdata = numpy.copy(self.data)
        limdata[limdata < seedlim] = 0

        #take random from nonzero of thresh data
        rng = numpy.random.default_rng(seed=1932)
        
        t_start = datetime.now()
        nonzero = numpy.array(limdata.nonzero()).T
        # print(self.threshed_data.shape)
        # print(nonzero.shape)
        
        seeds = rng.choice(nonzero, n_seed_points, replace=False)
        # seeds = nonzero
        # seeds = int(numpy.random.rand(n_seed_points, 3)*self.threshed_data.shape)
        
        self.t_rand = (datetime.now()-t_start)    
        print("Point generation: ", self.t_rand)    
        t_start = datetime.now() 

        weights = self.threshed_data[tuple(seeds.T)]
        weights = (weights-weights.min())/(weights.max() - weights.min())

        self.seed_points = sample_elimination(points = seeds, target=self.n_points, mindist=0, maxdist=0, weights=weights, params = params)        

        self.t_sample = (datetime.now()-t_start)    
        print("Sample Elimination: ", self.t_sample)   

        threshed_volume = pyvista.ImageData(pyvista.wrap(self.threshed_data))
        threshed_volume.spacing = self.spacing

        self.plotter.clear()
        self.plotter.add_volume(threshed_volume, opacity = [0,0.05], shade=False)
        self.plotter.add_points(seeds* self.spacing, opacity = 0.5)        
        self.plotter.add_points(self.seed_points * self.spacing, color = 'red')        
        self.plotter.reset_camera()

    @QtCore.Slot()
    def tesselateData(self):
        
        t_start = datetime.now() 
        self.tesselated_data = tesselate(mask=self.stencil,seeds=numpy.ascontiguousarray(self.seed_points))
        self.tesselated_volume = pyvista.wrap(self.tesselated_data)
        self.tesselated_volume.spacing = self.spacing
        self.t_tess = (datetime.now()-t_start)    
        print("Tessellation: ", self.t_tess)    
        self.plotter.clear()
        self.plotter.add_volume(self.tesselated_volume, opacity = [0,0.1], shade=False)
        self.plotter.add_points(self.seed_points* self.spacing, opacity = 0.5)        
        self.plotter.reset_camera()

    @QtCore.Slot()
    def createMesh(self):
        t_start = datetime.now()
        data = pyvista.wrap(self.tesselated_volume)
        grid = pyvista.create_grid(data)
        # grid.spacing = spacing
        data = grid.sample(data, categorical=True, progress_bar=True)
        # self.mesh = data.contour_labeled(n_labels = self.n_points, smoothing = True)
        data = data.pack_labels()
        # print(numpy.unique(data.active_scalars))
        self.mesh = data.contour_labeled(n_labels = self.n_points, smoothing = True, progress_bar=True)
        # data = data.decimate_pro(0.5)

        self.mesh = self.mesh.connectivity('largest')
        self.t_mesh = (datetime.now()-t_start)    
        print("Meshing: ", self.t_mesh)    
        t_total = self.t_rand + self.t_sample + self.t_tess + self.t_mesh
        print("Performance")    
        print("-------------")    
        print("Point generation: ", self.t_rand)   
        print("ample Elimination: ", self.t_sample)   
        print("Tessellation: ", self.t_tess)   
        print("Meshing: ", self.t_mesh)   
        print("-------------")    
        print("Total Time: ", t_total)    
        # mesh = mesh.scale((0.5,0.5,1))

        # for thing in self.mesh.active_scalars:

        # print(self.mesh["BoundaryLabels", 4])

        self.plotter.clear()

        # self.plotter.add_mesh(self.mesh, opacity = 0.1)
        self.plotter.add_mesh(self.mesh, opacity = 0.1)
        self.plotter.remove_scalar_bar()
        # self.plotter.add_points(self.seed_points)        
        self.plotter.reset_camera()

    
    @QtCore.Slot()
    def saveMesh(self):   
        fname = QtWidgets.QFileDialog.getSaveFileName(filter="*.stl")[0]
        if fname is not None:    
            self.mesh.save(fname)
        
    

def get_neigbours(x, y, dist):
    # if weight != 0: maxdist = maxdist/(weight)
    # else: maxdist = numpy.inf
    dists = numpy.linalg.norm(x - y,axis=1)
    neighbor_inds = numpy.asarray(dists<2*dist).nonzero()
    # print("point ", y, "neighbors", neighbor_inds)
    # print(neighbor_inds)
    return neighbor_inds

def get_weight(x,y,dmax,  n=1, m=3, alpha = 8, beta = 0.65, gamma = 1.5):
    dmin = dmax * (1-(n/m)**gamma) * beta
    dist = numpy.linalg.norm(x - y)
    if(dist < 2*dmin):dist = dmin
    if(dist > 2*dmax):dist = dmax
    weight = (1-(dist/(2*dmax)))**alpha
    return weight

def linear_dist_interpolation(weight, min, max, params):
    return (min + weight * (max-min))

def exp_dist_interpolation(weight, min, max, params):
    test = (min + (weight**5)  * (max-min))
    return test

def pwl_dist_interpolation(weight, min, max, params):
    x = params[0]
    y = params[1]
    res = y[0]
    for i in range(0, len(x)-1):
        if(weight >= x[i] and weight <= x[i+1]): res = y[i] + (y[i+1] - y[i])*(weight - x[i])/(x[i+1] - x[i])
    
    # test = (min + res  * (max-min))
    return res

def sample_elimination(points, target, mindist, maxdist, weights=None, dist_interpolation=pwl_dist_interpolation, params=None):
    result = numpy.copy(points)
    # print(result.shape)
    m = points.shape[0]
    n = target
    if weights is None:
        weights = numpy.ones(result.shape[0])
    weights = weights[:,numpy.newaxis]
    # print(weights)
    result = numpy.concatenate((result, numpy.array(weights)), axis=1)
    # print(dist_interpolation(0.5, mindist, maxdist, params))

    # for point in result:
    #     print(point, get_neigbours(result[:,0:3], point[0:3], maxdist)[0].shape[0], len(get_neigbours(result[:,0:3], point[0:3], maxdist)[0]) / point[3])
    # cost = numpy.array([sum([get_weight(neighbor[0:3], point[0:3], maxdist) for neighbor in result[get_neigbours(result[:,0:3], point[0:3], maxdist, point[3])[0]]]) for point in result])
    cost = numpy.array([sum([get_weight(neighbor[0:3], point[0:3], dist_interpolation(neighbor[3], mindist, maxdist, params), n,m) for neighbor in result[get_neigbours(result[:,0:3], point[0:3], dist_interpolation(point[3], mindist, maxdist, params))[0]]]) for point in result])
    # cost = numpy.array([len(get_neigbours(result[:,0:3], point[0:3], maxdist)[0]) / point[3] for point in result])

    # print(cost)

    result[:,3] = cost


    while result.shape[0] > target:
        # print(result)
        # print(result.shape)
        candidate_ind = numpy.argmax(result[:,3])
        # print(result[candidate_ind])
        
        candidate = result[candidate_ind]
        candidate_neighbors_inds = get_neigbours(result, candidate, dist_interpolation(candidate[3], mindist, maxdist, params))[0]
        # print("candidate ", candidate_ind, "neighbors ", len(candidate_neighbors_inds))
        candidate_neighbors = result[candidate_neighbors_inds]
        # print(result[candidate_neighbors_inds][:,3])
        for ind in candidate_neighbors_inds:            
            result[ind,3] -= get_weight(result[ind, 0:3], candidate[0:3], dist_interpolation(candidate[3], mindist, maxdist, params), n,m)
        # print(result)
        result = numpy.delete(result, candidate_ind,0)
    pass

    return result[:,0:3]


def sampleAny(path, density = 0.1, thresholdP = 0.1, maxpoints = 100):
    rd = pyvista.get_reader(path)
    data = rd.read()
    points = numpy.reshape(data.active_scalars, data.dimensions, order = "F")
    points = (points-points.min())/(points.max() - points.min())
    return sampleGrid(points, density = density, thresholdP= thresholdP, maxpoints = maxpoints)

def sampleOpenScivis(path, density = 0.1, thresholdP = 0.1, maxpoints = 100, exp = 1/4):
    type = numpy.dtype(path.split('/')[-1].split('_')[-1].split('.')[0])
    shape = path.split('/')[-1].split('_')[-2]
    shape = [
        int(path.split('/')[-1].split('_')[-2].split('x')[0]),
        int(path.split('/')[-1].split('_')[-2].split('x')[1]),
        int(path.split('/')[-1].split('_')[-2].split('x')[2])]
    
    return sampleRAW(path,shape,type, density, thresholdP, maxpoints, exp)

def sampleRAW(path, shape, dtype, density = 0.1, thresholdP = 0.1, maxpoints = 100, exp = 1/4):
    
    f = open(path, 'rb')  
    d = f.read()
    f.close()
    data = numpy.frombuffer(d,  dtype)
    data = data.reshape(shape, order='F')    
    data = (data-data.min())/(data.max() - data.min())

    return sampleGrid(data, density=density, thresholdP=thresholdP, maxpoints=maxpoints, exp = exp)

def sampleDicom(path, density = 0.1, thresholdP = 0.1, maxpoints = 10000):    
    rd = pyvista.DICOMReader(path)
    data = rd.read()
    points = numpy.reshape(data.active_scalars, data.dimensions, order = "F")
    points = (points-points.min())/(points.max() - points.min())
    return sampleGrid(points, density, thresholdP, maxpoints = 100)

# def load_data()

def cleanVolume(vol):

    volData = pyvista.wrap(vol)
    volData = volData.image_dilate_erode(erode_value=0, dilate_value=1)
    volData = volData.image_dilate_erode(erode_value=1, dilate_value=0)
    volData = volData.image_dilate_erode(erode_value=1, dilate_value=0)
    volData = volData.image_dilate_erode(erode_value=0, dilate_value=1)

    return numpy.reshape(volData.active_scalars, vol.shape, order='F')

def load_raw(path):
    dtype = numpy.dtype(path.split('/')[-1].split('_')[-1].split('.')[0])
    shape = [
        int(path.split('/')[-1].split('_')[-2].split('x')[0]),
        int(path.split('/')[-1].split('_')[-2].split('x')[1]),
        int(path.split('/')[-1].split('_')[-2].split('x')[2])]
    f = open(path, 'rb')  
    d = f.read()
    f.close()
    data = numpy.frombuffer(d,  dtype)
    data = data.reshape(shape, order='F')    
    # data = (data-data.min())/(data.max() - data.min())

    return data

def tesselate(mask, seeds):

    inds = numpy.transpose(numpy.nonzero(mask))

    regions = numpy.zeros(inds.shape[0])
    min_dists = numpy.ones(inds.shape[0])
    min_dists = min_dists * numpy.inf
    threadsperblock = 100
    blockspergrid_x = math.ceil(min_dists.size/ threadsperblock ) 

    polyhedron = voronoi_gpu.get_btch()



    # start = datetime.now()
    voronoi_gpu.multi_seed_min_polyhedral_dist_3point[blockspergrid_x, threadsperblock](numpy.ascontiguousarray(inds), numpy.ascontiguousarray(min_dists), numpy.ascontiguousarray(regions), seeds, polyhedron)
    # t2 = datetime.now()
    # print("t2 ", t2 - start)

    regs = numpy.zeros_like(mask)

    regs[tuple(inds.T)] = regions   

    # regs = regs +1

    return regs



if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    
    widget = Gui()
    widget.resize(800, 600)
    widget.move(100, 100)
    widget.setWindowTitle("Squishicalization")
    widget.show()
    widget.raise_()
    widget.activateWindow()

    sys.exit(app.exec())


