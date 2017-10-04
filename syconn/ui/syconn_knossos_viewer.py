# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

from PythonQt import QtGui, Qt, QtCore
import KnossosModule
import sys
sys.dont_write_bytecode = True

# need to specify paths to syconnfs code base manually if not in path
# make this a GUI setting
# TODO: get from config
syconnfs_path = '/wholebrain/scratch/areaxfs/'

from ..representations import super_segmentation as ss


class main_class(QtGui.QDialog):
    def __init__(self, parent=KnossosModule.knossos_global_mainwindow):
        #Qt.QApplication.processEvents()
        super(main_class, self).__init__(parent, Qt.Qt.WA_DeleteOnClose)
        try:
            exec(KnossosModule.scripting.getInstanceInContainerStr(__name__)\
                 + " = self")
        except KeyError:
            # Allow running from __main__ context
            pass
        #self.start_logging()

        self.ssv_selected1 = 0

        self.init_syconn()
        self.build_gui()

    def init_syconn(self):

        self.ssd = ss.SuperSegmentationDataset(syconnfs_path)

        return

    def populate_ssv_list(self):
        all_ssv_ids = self.get_list_of_all_ssv_ids()
        #print('list of all here')
        #print(len(all_ssv_ids))
        for ssv_id in all_ssv_ids:
            #print(ssv_id)
            item = QtGui.QStandardItem(str(ssv_id))
            self.ssv_item_model.appendRow(item)

        self.ssv_selector.setModel(self.ssv_item_model)
        return

    def on_ssv_selector_changed(self, index):
        self.ssv_selected1 = int(self.ssv_selector.model().itemData(index)[0])
        #current, previous
        #print('selected: ' + str(self.ssv_selector.model().itemData(index)[0]))
        #ssv = self.get_ssv(self.ssv_selected1)
        #self.ssv_to_knossos(ssv)

        return

    def build_gui(self):
        self.setWindowFlags(Qt.Qt.Window)
        layout = QtGui.QGridLayout()
        layout.setSpacing(10)

        # Window layout
        #layout = QtGui.QVBoxLayout()
        self.setLayout(layout)

        self.show_button = QtGui.QPushButton('Show neurite')
        self.clear_knossos_view_button = QtGui.QPushButton('Clear view')

        self.ssv_selector = QtGui.QListView()
        self.ssv_selector.setUniformItemSizes(True) # better performance
        self.ssv_item_model = QtGui.QStandardItemModel(self.ssv_selector)

        self.direct_ssv_id_input = QtGui.QLineEdit()
        self.direct_ssv_id_input.setValidator(QtGui.QIntValidator())
        self.direct_ssv_id_input.setMaxLength(16)

        #self.ssv_selection_model =
        # QtGui.QItemSelectionModel(self.ssv_select_model)

        #self.selectionModel =
        # self.ssv_selector.selectionModel(self.ssv_selector)
        #self.ssv_selector.setSelectionModel(self.ssv_selection_model)
        #print('selection model: ' + str(self.ssv_selector.selectionModel()))

        self.ssv_selector.clicked.connect(self.on_ssv_selector_changed)

        self.populate_ssv_list()
        print('Connected to SyConnFS backend')

        layout.addWidget(self.direct_ssv_id_input, 1, 0, 1, 2)
        layout.addWidget(self.ssv_selector, 2, 0, 1, 2)
        layout.addWidget(self.show_button, 3, 0, 1, 1)
        layout.addWidget(self.clear_knossos_view_button, 3, 1, 1, 1)

        #self.ssv_select_model.itemChanged.connect(self.on_ssv_selector_changed)
        #self.selectionModel.selectionChanged.connect(self.on_ssv_selector_changed)

        self.show_button.clicked.connect(self.show_button_clicked)
        self.clear_knossos_view_button.clicked.connect(self.clear_knossos_view_button_clicked)

        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('SyConn Viewer v1')
        self.show()
        #self.merge_button = QtGui.QPushButton('Merge')
        #self.split_button = QtGui.QPushButton('Split')


        #self.bad_button = QtGui.QPushButton('Bad SV')
        #self.graph_split_button = QtGui.QPushButton('Graph split')
        #self.add_selected_sv_button = QtGui.QPushButton('Add selected SV')
        #self.mode_combo = QtGui.QComboBox()
        #self.stop_button = QtGui.QPushButton('Stop')
        #self.undo_button = QtGui.QPushButton('Undo')
        #self.redo_button = QtGui.QPushButton('Redo')



        #self.skip_task_line_edit = QtGui.QLineEdit('Skip reason')
        #self.server_line_edit = QtGui.QLineEdit()

        #self.password_line_edit = QtGui.QLineEdit()
        #  set echo mode to QLineEdit::PasswordEchoOnEdit

        #self.gui_auto_agglo_line_edit = QtGui.QLineEdit()
        #self.gui_auto_agglo_line_edit.setText('0')

    def show_button_clicked(self):
        try:
            self.ssv_selected1 = int(self.direct_ssv_id_input.text)
        except:
            pass

        if self.ssv_selected1:
        #ssv_id = self.ssv_selected1
            ssv = self.get_ssv(self.ssv_selected1)
            self.ssv_to_knossos(ssv)

        return

    def clear_knossos_view_button_clicked(self):

        # delete all existing objects in mergelist
        all_objects = KnossosModule.segmentation.objects()
        [KnossosModule.segmentation.removeObject(obj) for obj in all_objects]

        # iterate over all tree in knossos and delete
        trees = KnossosModule.skeleton.trees()
        ids_in_k = set([tree.tree_id() for tree in trees])
        [KnossosModule.skeleton.delete_tree(sv_id) for sv_id in ids_in_k]

        return

    def ssv_to_knossos(self, ssv):

        self.clear_knossos_view_button_clicked()

        # to mergelist
        KnossosModule.segmentation.createObject(ssv.id, ssv.sv_ids[0], (1,1,1))
        #KnossosModule.segmentation.selectObject(ssv.id)
        # query object should be red
        KnossosModule.segmentation.changeColor(ssv.id, QtGui.QColor(255, 0, 0, 255))

        # one could cache this, not necessary to rebuild at every step
        for sv_id in ssv.sv_ids:
            try:
                KnossosModule.segmentation.addSubobject(ssv.id, sv_id)
            except:
                pass

        KnossosModule.segmentation.selectObject(ssv.id)

        KnossosModule.segmentation.setRenderOnlySelectedObjs(True)

        KnossosModule.skeleton.add_tree_mesh(1, ssv.mesh[1], [], ssv.mesh[0],
                                             [],
                                             KnossosModule.GL_TRIANGLES,
                                             False)
        KnossosModule.skeleton.set_tree_color(1, QtGui.QColor(255, 0, 0, 128))

        KnossosModule.skeleton.add_tree_mesh(2, ssv.mi_mesh[1], [],
                                             ssv.mi_mesh[0], [],
                                             KnossosModule.GL_TRIANGLES,
                                             False)
        KnossosModule.skeleton.set_tree_color(2, QtGui.QColor(0, 0, 255, 255))

        return

    def get_list_of_all_ssv_ids(self):
        """
        Fetches a list of all ssvs from syconn and adds them to the list. 
        
        Returns
        -------

        """

        return self.ssd.ssv_ids

    def get_ssv(self, ssv_id):

        ssv = self.ssd.get_super_segmentation_object(ssv_id)
        ssv.load_attr_dict()
        return ssv

if __name__=='__main__':
    A = main_class()