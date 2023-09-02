from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QPixmap,QMovie
import pyvista as pv
from pyvistaqt import QtInteractor
import os
import numpy as np
import predict
import llm_analysis_test
import robotarmcontrol
import threading
import time
class MainWindow(QMainWindow):
    '''main program: display interface, control camera, YOLO model, llm model, robot arm'''
    def __init__(self):
        super().__init__()
        #create model for object detection
        self.model = predict.Model()
        self.llmmodel = llm_analysis_test.LanguageModel()
        self.response = ""#response from gpt
        self.cls = ""
        self.message = ""
        #create robot arm object
        ROBOT_IP  = "192.168.8.130"
        self.robot = robotarmcontrol.URRobot(ROBOT_IP)
        ##########################Panel Creation#########################################
        self.setWindowTitle("Main Panel")
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setGeometry(50,50, 1200,800)
        # Create 3D point cloud view
        absolute_path = os.path.dirname(__file__)
        self.point_cloud_path = os.path.join(absolute_path, "source/results.ply")
        #self.gif_path = os.path.join(os.path.dirname(__file__),"source/loading_animation.gif")
        #self.segment_img_path = [os.path.join(absolute_path, "segmented_images/object"),".jpg"]
        point_cloud_view = QLabel("3D Point Cloud View")
        point_cloud_view.setAlignment(Qt.AlignCenter)
        self.plotter = QtInteractor(self)
        self.plotter.set_background('black')

        # Create the "Load Point Cloud" button
        self.load_button = QPushButton('Capture/Update Point Cloud', self)
        self.load_button.clicked.connect(self.load_point_cloud)
        #create a button for resetting camera angle
        self.reset_camera_button = QPushButton('Reset Camera Angle', self)
        self.reset_camera_button.clicked.connect(self.reset_camera_angle)
        #add them to layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.reset_camera_button)

        # Create a layout for the main window and add the PyVista widget and the button layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(point_cloud_view)
        self.layout.addWidget(self.plotter.interactor)
        self.layout.addLayout(button_layout) 
        #create point cloud view end =================================================================
        # Create image view
        self.image_view = QLabel("Image View")
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_Widgets = QLabel()
        self.image_Widgets.setFixedSize(point_cloud_view.size())
        self.image_Widgets.setAlignment(Qt.AlignTop)
        self.image_Widgets.setScaledContents = True
        #create images area ends =============================================================
        label_width = 200
        label_height = 200
        #create bubble chatbox

        #create loading animation
        #self.loading_gif = QMovie(self.gif_path)
        #self.loading_gif.setScaledSize(QSize(label_width,label_height))
        #self.loading_gif.start()
        #self.loading = QLabel()
        #self.loading.setMovie(self.loading_gif)
        # Create message list
        self.message_list = QListWidget()
        self.message_list.setWordWrap(True)
        self.message_list.setFlow(QListWidget.TopToBottom)
        self.message_list.setUpdatesEnabled(True)
        # Create chat log
        self.chat_log_layout = QVBoxLayout()
        self.chat_log_layout.addWidget(self.message_list)

        # Create message input and send button
        self.message_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.display_new_chats)
        self.message_layout.addWidget(self.message_input)
        self.message_layout.addWidget(self.send_button)

        #Create confirm button for confirming object chose
        self.confirm_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirm_object)
        self.object_selected = QLineEdit()
        self.object_selected.setReadOnly(1)
        self.confirm_layout.addWidget(self.object_selected)
        self.confirm_layout.addWidget(self.confirm_button)
        #create chatroom ends ==========================================================
        
        # Create overall layouts
        view_overall_layout = QHBoxLayout()
        view_layout2 = QVBoxLayout()
        view_layout2.addWidget(self.image_view)
        view_layout2.addWidget(self.image_Widgets)
        view_overall_layout.addLayout(self.layout,stretch = 1)
        view_overall_layout.addLayout(view_layout2,stretch = 1)
 
        self.chat_layout = QVBoxLayout()
        self.chat_layout.addLayout(self.chat_log_layout)
        self.chat_layout.addLayout(self.message_layout)
        self.chat_layout.addLayout(self.confirm_layout)
        main_layout = QVBoxLayout()
        main_layout.addLayout(view_overall_layout,stretch = 1)
        main_layout.addLayout(self.chat_layout,stretch = 1)

        # Set central widget layout
        central_widget.setLayout(main_layout)
        #create overall layout ends =======================================
    #########################end of creating panel############################################################
        
    '''update the point cloud displayed at top left corner'''
    def load_point_cloud(self):
        # Open a file dialog to select a point cloud file
        self.robot.move_to_coord(self.robot.init_pose,False)
        #time.sleep(3)
        self.model.init_pointcloud()
        filename = self.point_cloud_path
        if filename:
            # Load the point cloud file
            point_cloud = pv.read(filename)
            # Add the point cloud to the PyVista widget
            self.plotter.clear()
            self.plotter.add_points(point_cloud,rgb = True)
            self.plotter.camera.tight(adjust_render_window = False, view = 'yx')
            self.plotter.camera.roll -=90
            self.plotter.reset_camera()
            #call functions to update image
            self.display_images()

    '''Display updated image at top right corner'''
    def display_images(self):
        pixmap = QPixmap(self.model.out_image_path).scaled(800,500,Qt.KeepAspectRatio)
        self.image_Widgets.setPixmap(pixmap)

    '''reset camera angle'''
    def reset_camera_angle(self):
        self.plotter.camera.tight(adjust_render_window = False, view = 'yx')
        self.plotter.camera.roll -=90
        self.plotter.reset_camera()


    '''send prompt to gpt and display message'''
    def send_prompt(self, message):
        self.response = None  
        object_list = self.model.get_original_object_list() 
        self.response = self.llmmodel.answer_prompt(message, object_list)
        row = self.message_list.count() - 1
        self.message_list.takeItem(row)
        item = QListWidgetItem(f"{self.response}")
        self.message_list.insertItem(row, item)
        ##############################################
        found_obj = False
        object = ""
        self.response = self.response.lower()
        for obj in object_list:
            if obj in self.response:
                if not found_obj:
                    found_obj = True
                    object = obj
                else:
                    if obj != object:
                        #more than one object, ignore this response
                        found_obj = False
                        return 
                    else:
                        continue
        if object != "":
            self.object_selected.setText(object)

    '''Display chats onto chat room'''
    def display_chats_thread(self):
        self.message = self.message_input.text()
        self.message_input.setText("")
        if self.message:
            item = QListWidgetItem(f"You: {self.message}")
            item.setTextAlignment(Qt.AlignRight)
            self.message_list.addItem(item)
            item = QListWidgetItem("...GPT writing...")
            self.message_list.addItem(item)
            self.message_list.scrollToBottom()
            self.message_list.repaint()
            QCoreApplication.processEvents()

    '''Display chats onto chat room'''
    def display_new_chats(self):
            #if contain message
            self.display_chats_thread()
            self.send_prompt(self.message)

    '''when user confirm, robot arm will go to grab the object and return to original position''' 
    def confirm_object(self):
        object =  self.object_selected.text()
        self.object_selected.setText("")
        if object != "":
            coordinates,vertical = self.model.get_object_3D_coordinates(object)
            if not np.isnan(coordinates[0]):
                if vertical:
                    rotation = [2.87,1.2,0.067]
                else:
                    rotation = [-2.92, 1.03, -0.047]
                moved = self.robot.move_to_coord([coordinates[0]/1000,coordinates[1]/1000, 0.25, rotation[0], rotation[1], rotation[2]])
                if moved:
                    print("grabbing object")
                    #self.robot.move_to_table()
                    self.robot.grab_object()
                    #re init the point cloud
                    self.load_point_cloud()
            else:
                print("coordinates unknown")

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exit(app.exec_())