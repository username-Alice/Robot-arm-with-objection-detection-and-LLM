import zivid
from PIL import Image
from datetime import timedelta
import os
import numpy as np
class Camera():
    '''capture 3D point cloud and 2D image, help predict.py extract 3D coordinates(w.r.t to base and camera) as well'''
    def __init__(self):
        super(Camera).__init__() 
        app = zivid.Application()
        #default setting
        
        ##########################################################################################
        #default = os.path.join(absolute_path, "./Zivid_sample_files/FileCameraZividOnePlusL.zfc")
        #self.camera = app.create_file_camera(default)
        ##########################################################################################

        #when have connected camera
        self.camera = app.connect_camera()
        
        #suggest_settings_parameters = zivid.capture_assistant.SuggestSettingsParameters() 
        #self.settings = zivid.capture_assistant.suggest_settings(self.camera,suggest_settings_parameters)
        #custom setting
        self.settings = zivid.Settings()
        self.settings.acquisitions.append(zivid.Settings.Acquisition())
        self.settings.acquisitions[0].brightness = 1.8
        self.settings.acquisitions[0].gain = 1.0
        self.settings.acquisitions[0].exposure_time = timedelta(microseconds=10000)
        self.settings.processing.color.gamma = 0.6
        self.settings.processing.color.experimental.mode = "toneMapping"
        #=================================================================
        absolute_path = os.path.dirname(__file__)
        self.cap_path = os.path.join(absolute_path, "source/results.ply")
        self.twoD_img_path = os.path.join(absolute_path, "source/camera_picture.jpg")
        self.yaml_file_path = os.path.join(absolute_path, "source/handEyeTransform.yaml")
        self.transform_camera_to_base = np.array(zivid.Matrix4x4(self.yaml_file_path))
    
    '''control camera to get new images'''
    def capture_objects(self):
        with self.camera.capture(self.settings) as self.frame:
            self.frame.save(self.cap_path)
            self.Point_cloud = self.frame.point_cloud()
            self.rgba = self.Point_cloud.copy_data("rgba")
            self.xyz = self.Point_cloud.copy_data("xyz")
            im = Image.fromarray(self.rgba).convert('RGB')
            im.save(self.twoD_img_path)
            return im,self.xyz
        
    '''return a center point of bounding box, helper function'''
    '''def get_coord_from_2Dim(self,box):
        #box is in xyxy format
        x_center = int((box[0]+box[2])/2)
        y_center = int((box[1]+box[3])/2)
        print(x_center,",", y_center)
        coord = self.xyz[y_center][x_center]
        #print("new_centroid_testing = ",self.testing_new_centroid(box))
        return coord'''

    def get_3D_from_2D(self,twoD):
        x = twoD[0]
        y = twoD[1]
        return self.xyz[y][x]

    '''def is_nan(self,two_D_coordinates):
        x = two_D_coordinates[0]
        y = two_D_coordinates[1]
        return self.xyz[y][x][0] == np.nan or self.xyz[y][x][1]  == np.nan '''
    
    def calculate_centroid(self, box):
        #box in xyxy format
        xyz = self.xyz[int(box[1]):int(box[3]),int(box[0]):int(box[2])][:]
        if xyz != np.array([]):
            mean = np.nanmean(xyz, axis = (0,1))
            print("centroid = ",mean)
        return mean

    '''transform a bounding box to a 3D coordinates w.r.t. robot arm base'''
    def transform_box_to_robot_coord(self,xyz):
        #xyz = self.get_coord_from_2Dim(box)
        #xyz = self.xyz[y_center][x_center]
        point_in_camera_frame = np.array(
        [
        xyz[0],
        xyz[1],
        xyz[2],
        1,
        ]
        )
        print(f"Point coordinates in camera frame: {point_in_camera_frame[0:3]}")
        print("Transforming (picking) point from camera to robot base frame")
        point_in_base_frame = np.matmul(self.transform_camera_to_base, point_in_camera_frame)
        print(f"Point coordinates in robot base frame: {point_in_base_frame[0:3]}")
        return point_in_base_frame
        
#for testing purpose
"""img, point_cloud = capture_objects()
img.show()
print(point_cloud)"""
#get point cloud and image from zdf file
 
