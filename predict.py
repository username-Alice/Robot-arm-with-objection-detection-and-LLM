#model
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import camera
class Model(YOLO):
    '''contain object detection model, also help to identify the actual 3D coordinates of the object, directly communicate with gui'''
    def __init__(self):
        super().__init__()
        absolute_path = os.path.dirname(__file__)
        #self.model = YOLO(os.path.join(absolute_path,"source/model/yolov8n.pt"))
        self.model = YOLO(os.path.join(absolute_path,"source/model/yolov8x-seg.pt")) 
        self.Camera = camera.Camera()
        #images paths
        self.image_path = os.path.join(absolute_path, "source/camera_picture.jpg")
        self.out_image_path = os.path.join(absolute_path, 'source/predictions.jpg')
        self.segment_img_path = [os.path.join(absolute_path, "segmented_images/object"),".jpg"]
        self.results = ""
        self.point_cloud = None
        #
        #image segmentation
        '''for i in range(len(self.results[0].boxes.xyxy)):
            self.box = []
            for j in range(4):
                tensorbox = self.results[0].boxes[i].xyxy[0].tolist()
                self.box.append(tensorbox[j])
            temp = original_image.crop((self.box))
            temp.save(self.segment_img_path[0]+str(i)+self.segment_img_path[1])'''
        #segmentation ends
    
    '''initialise new pointcloud, save new RGB image'''
    def init_pointcloud(self):
        #this function will be used to init the point cloud data
        image, self.point_cloud = self.Camera.capture_objects()
        image.save(self.image_path)
        original_image = Image.open(self.image_path)
        self.results = self.model(original_image)
        #this output the entire image
        imageRGB = cv2.cvtColor(self.results[0].plot(), cv2.COLOR_BGR2RGB)
        results_plotted = Image.fromarray(imageRGB)
        results_plotted.save(self.out_image_path)
        
    '''get detected object list'''
    def get_original_object_list(self):
        if self.results == "":
            return []
        names = self.results[0].names
        class_labels = self.results[0].boxes.cls
        class_labels_actual = []
        for i in range(len(class_labels)):
            class_labels_actual.append(names.get(int(class_labels[i])))
        return class_labels_actual
    
    '''get unique object list (no object names repeated)'''
    def get_object_list(self):
        #return a list of unique prediction results
        class_labels_actual = self.get_original_object_list()
        
        #extract a list of class label to later feed into gpt
        unique_class_labels = np.unique(np.array(class_labels_actual))
        return unique_class_labels
    
    '''get original object list length'''
    def get_list_len(self):
        return len(self.results[0].boxes.cls)
    
    '''return true if bounding box is vertical, use for gripper orientation modification'''
    def vertical_bounding_box(self, box):
        if abs(box[0]-box[2]) < abs(box[1] - box[3]):
            return True
        return False

    '''given object name, find the 3D coord w.r.t. robot arm base'''
    def get_object_3D_coordinates(self, object):
        #get 2D coordinates based on object ID
        array = np.array(self.get_original_object_list())
        index = np.where(array == object)[0]
        if (len(index) >= 1):
            i = index[0] ##get the first object box 
        else:
            print("WARNING- No such object")
            return []
        
        box = self.results[0].boxes.xyxy[i]
        is_vertical = self.vertical_bounding_box(box)
        #map it to 3D coordinates
        coord_from_image = self.get_coordinates_from_image(i)
        xyz = self.Camera.calculate_centroid(box)
        if not np.isnan(coord_from_image[0]):
            print("image coordinates")
            #coord_from_image[0] += 10
            #coord_from_image[1] += 30
            coordinates= self.Camera.transform_box_to_robot_coord(coord_from_image)
        #coordinates = self.Camera.transform_box_to_robot_coord(xyz)
        else:
            coordinates = self.Camera.transform_box_to_robot_coord(xyz)
            
        print("final coordinates = ", coordinates)
        return coordinates,is_vertical

    '''an attempt to use segmented image to find centroid'''
    def get_coordinates_from_image(self,index):
        if(self.results[0].masks is not None):
            # Convert mask to single channel image
            mask_raw = self.results[0].masks[index].cpu().data.numpy().transpose(1, 2, 0)
            # Convert single channel grayscale to 3 channel image
            mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))
            # Get the size of the original image (height, width, channels)
            h2, w2, c2 = self.results[0].orig_img.shape
            # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
            mask = cv2.resize(mask_3channel, (w2, h2))
            # Define range of brightness in HSV
            lower_black = np.array([0,0,0])
            upper_black = np.array([0,0,1])
            # Create a mask. Threshold the HSV image to get everything black
            mask = cv2.inRange(mask, lower_black, upper_black)
            # Invert the mask to get everything but black
            mask = cv2.bitwise_not(mask)
            # Find contours:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Draw contours:
            cv2.drawContours(mask, contours, 0, (0, 255, 0), 2)

            # Calculate image moments of the detected contour
            M = cv2.moments(contours[0])

            # Draw a circle based centered at centroid coordinates
            cv2.circle(mask, (round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])), 5, (0, 255, 0), -1)
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(mask,[box],0,(0,0,255),2)
            # Show the masked part of the image
            cv2.imwrite("source/masked.jpg", mask)
            twoD = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])
            return self.Camera.get_3D_from_2D(twoD)
        else:
            print("WARNING: NO Objects is detected")
            return [np.nan,np.nan,np.nan]