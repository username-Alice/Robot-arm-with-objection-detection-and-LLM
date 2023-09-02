import numpy as np
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl
import time
from gripper import Gripper
'''control robot arm and control gripper'''
class URRobot:
    def __init__(self, robot_ip):
        print(f"[INFO] Connecting UR robot ...")
        self.rtde_receive = RTDEReceive(robot_ip, rt_priority=90)
        self.rtde_control = RTDEControl(robot_ip, rt_priority=85)
        self.init_pose = [-0.41, -0.35, 0.30, -2.92, 1.03, -0.047]
        ##### define safe area #########################################
        self.LeftTop = [-0.84, -0.18, 0.3,  -2.92, 1.03, -0.047]
        self.RightTop = [-0.50, 0.21, 0.3,  -2.92, 1.03, -0.047]
        self.RightBottom = [-0.14, -0.17, 0.3, -2.92, 1.03, -0.047]
        self.LeftBottom = [-0.5, -0.53, 0.3,  -2.92, 1.03, -0.047]

        #equations of rectanglar safe area (calculate whether arm lies within safe area)
        #Left Bottom to Right Bottom
        self.m_b = (self.LeftBottom[1]-self.RightBottom[1])/(self.LeftBottom[0] - self.RightBottom[0]);self.c_b = self.LeftBottom[1] - self.m_b*self.LeftBottom[0]
        #Left bottom to Left Top
        self.m_l = (self.LeftTop[1]-self.LeftBottom[1])/(self.LeftTop[0] - self.LeftBottom[0]);self.c_l = self.LeftTop[1] - self.m_l*self.LeftTop[0]
        #Left Top to Right Top
        self.m_t = (self.LeftTop[1]-self.RightTop[1])/(self.LeftTop[0] - self.RightTop[0]);self.c_t = self.RightTop[1] - self.m_t*self.RightTop[0]
        #right Top to Right bottom
        self.m_r = (self.RightBottom[1]-self.RightTop[1])/(self.RightBottom[0] - self.RightTop[0]);self.c_r = self.RightBottom[1] - self.m_r*self.RightBottom[0]

        ###### define speed, acceleration, etc###########################
        self.speed = 0.3
        self.acceleration = 0.1
        self.RobotRunning = True
        self.stop_flag = False
        self.offset = 0.01 #hand eye calibration error
        ############################create gripper#######################
        self.gripper = Gripper()
        self.gripper.connect(robot_ip)
        self.gripper.activate()
        self.gripper_force = 100


    @staticmethod
    def degree2rad(degree):
        return degree / (180 / np.pi)

    @staticmethod
    def rad2degree(rad):
        return rad * (180 / np.pi)


    def move_joints(self, joint_degrees, speed=1):
        joint_radians = list(map(URRobot.degree2rad, joint_degrees))
        self.rtde_control.moveJ(joint_radians, speed=speed, acceleration=self.acceleration)


    '''check whether the position is safe to reach'''
    def check_safe(self,tcp_pose):
        offset = 0.03
        x = tcp_pose[0]
        y = tcp_pose[1]
        within_top_right = (y < (max(self.m_t*x + self.c_t, self.m_r*x + self.c_r)-offset))
        within_bottom_left = (y > (min(self.m_b*x + self.c_b,self.m_l*x + self.c_l)+offset))
        if (not within_bottom_left or not within_top_right):
            #check x coordinates and y coordinates respectively
            print("WARNING: Out of Bound")
            self.move_to_coord(self.init_pose,False)
            return False
        return True
    
    '''move to target position given coordinates, check: check whether the position is within safe area '''
    def move_to_coord(self, coordinates,check = True):
        if check:
            if (not self.check_safe(coordinates)):
                return False
        self.rtde_control.moveL(coordinates,self.speed, self.acceleration,True)
        self.wait()
        return True

    '''move the robot arm down to the table - helper function for grab object'''
    def move_to_table(self):
        coordinates = self.get_tcp_pos()
        coordinates[2] = 0.0025
        #coordinates[2] = 0.005 #TEEMPTEEMMPTEPMEPP
        self.move_to_coord(coordinates, False)
    
    '''grab object function'''
    def grab_object(self):
        self.move_to_table()
        #try to grab the object
        self.gripper.move_and_wait_for_pos(255, 255, self.gripper_force)
        
        self.move_to_coord(self.init_pose, False)
        self.move_to_table()
        self.gripper.move_and_wait_for_pos(0, 255, self.gripper_force)
        self.move_to_coord(self.init_pose, False)

    '''testing function'''
    def move_in_safe_area(self): #for testing only
        self.move_to_coord(self.init_pose)
        self.move_to_coord(self.LeftTop)
        self.move_to_coord(self.RightTop)
        self.move_to_coord(self.RightBottom)
        self.move_to_coord(self.LeftBottom)

    '''return whether the arm is moving'''
    def is_moving(self):
        #return np.all(np.array(self.get_tcp_pos()) == np.array(self.get_target_tcp_pose()))
        return np.any(np.array(self.rtde_receive.getActualTCPSpeed()))
    
    '''wait until the arm reach destination'''
    def wait(self):
        time.sleep(1)
        while self.is_moving():
            time.sleep(0.1)

    '''get current x,y,z,rx,ry,rz position'''
    def get_tcp_pos(self):
        return self.rtde_receive.getActualTCPPose()
    
    '''get target position'''
    def get_target_tcp_pose(self):
        return self.rtde_receive.getTargetTCPPose()
    
    '''get joint position'''
    def get_joint_degrees(self):
        joint_radians = self.rtde_receive.getActualQ()
        joint_degrees = list(map(URRobot.rad2degree, joint_radians))
        return joint_degrees
