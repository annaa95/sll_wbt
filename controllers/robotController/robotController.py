# The Robot Controller

from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV
from numpy import linalg as LA
from collections.abc import Iterable
import pandas as pd

#from importlib.machinery import SourceFileLoader
#StateMachine = SourceFileLoader("StateMachine","/home/anna/Documenti/webots_projects/SingleLegLearner_extern/controllers/robotController/FiniteStateMachine.py").load_module()
from FiniteStateMachine import StateMachine
#from utilities import normalizeToRange
#from utilities import real_time_peak_detection
#from utilities import createFilt
#from tensorboardX import SummaryWriter

import math
import numpy as np
import scipy.signal as sig

class SLL_RobotController(RobotEmitterReceiverCSV):
    
    def __init__(self):
        super().__init__()        
        #print("init Robot")
        # ----robot geometric specs----
        self.l1 = 0.5 #m
        self.l2 = 0.5 #m
        self.Q = np.zeros(2) #motori
        self.t0 = self.robot.getTime()
        self.t_last_ev = self.robot.getTime()
        self.tmin = 0.2 #0.2s
        self.tmax = 5#00*self.timestep*1e-3 #6.4s
        self.y_min = 0.05
        self.cnt_thr =0.9
        self.deCnt_thr= 0.1
        self.contact = False
        self.states = ["TD", "LO", "Stance", "Flight"]
        
        # ----robot sensors init----
        self.pressure_sensor = self.robot.getDevice("pressure_reader")
        self.pressure_sensor.enable(self.timestep)
        self.pressure_sensor.setChannel(1)
        self.accelerometer = self.robot.getDevice("accelerometer")
        self.accelerometer.enable(self.timestep)
        self.gyroscope = self.robot.getDevice("gyro")
        self.gyroscope.enable(self.timestep)
        # ----robot actuators init----
        motors = ['hip_motor', 'knee_motor']
        self.motor=[]
        for name in motors:
            self.motor.append(self.robot.getDevice(name))
        self.encoders = [self.robot.getDevice("hip_encoder"), self.robot.getDevice("knee_encoder")]
        for i in range(2):
            self.encoders[i].enable(self.timestep)
        # ----robot other vars init----
        self.message = None
        #---- define the FSM for the locomotion controller---
        self.m = StateMachine()
        self.m.add_state("Flight_state", self.flight_state_transitions)
        self.m.add_state("Stance_state", self.stance_state_transitions)
        self.m.add_state("Failed_state", None, end_state=1)
        self.m.set_start("Flight_state")
        try:
            self.handler = self.m.handlers[self.m.startState]
        except:
            raise InitializationError("must call .set_start() before .run()")
        if not self.m.endStates:
            raise  InitializationError("at least one state must be an end_state")

    def create_message(self):
        TD = []
        for i in range(len(self.TouchSensors)):
            TD.append(self.TouchSensors[i].getValue())
            #print("TS"+str(i)+" value:", self.TouchSensors[i].getValue())
        self.message = TD 
        self.contacts = TD  
        return self.message
    
    def use_message_data(self, message):
        """
        The usage of message data is related to the specific state
        of the FSM we are using 
        """
        t =self.robot.getTime()-self.t0
        message = np.asarray(message)
        #print("Received Message:",message)
        if len(message)== 4 :
            hlb = message[0] #high level behavior
            if hlb == 'CC':
                pos = np.deg2rad(float(message[1]))*np.sin(2.0*np.pi*float(message[2])*t+float(message[3]))
                vel = np.deg2rad(float(message[1]))*float(message[2])*np.cos(2.0*np.pi*float(message[2])*t+float(message[3]))
                self.motor[0].setPosition(pos)
                self.motor[0].setVelocity(abs(vel))
                self.motor[1].setPosition(-pos)
                self.motor[1].setVelocity(abs(vel))    
            elif hlb == 'MFP':
                #message : alpha_td, omega, theta_0, y
                self.alphaTD = float(message[1])
                self.omega = float(message[2])
                self.theta0 = float(message[3])
                #print("contact reading:", self.contacts)
                l_square = self.l1**2+self.l2**2-2*self.l1*self.l2*np.cos(np.deg2rad(self.theta0))
                beta = math.acos((self.l1**2+l_square-self.l2**2)/(2*self.l1*np.sqrt(l_square)))
                self.q1_i = np.pi -np.deg2rad(self.theta0)
                self.q0_i = np.deg2rad(self.alphaTD)-beta

                (newState) = self.handler()
                if newState.upper() in self.m.endStates:
                    pass#print("reached ", newState) 
                else:
                    self.handler = self.m.handlers[newState.upper()]  
                #self.FSM(float(message[1]), float(message[2]), float(message[3]), t)
            else:
                print('unknown behavior: '+message[0])
        return 1

    def read_pressure_sensor(self):
        """
        This receiver uses the basic Webots receiver-handling code. The
        use_message_data() method should be implemented to actually use the
        data received from the supervisor.
        """
        if self.pressure_sensor.getQueueLength() > 0:
            # Receive and decode message from supervisor
            press_msg = self.pressure_sensor.getData().decode("utf-8")
            # Convert string message into a list
            press_msg = press_msg.split(",")
            press_msg = np.asarray(press_msg)

            if len(press_msg)== 1 :
                bodyPosY = press_msg
            self.pressure_sensor.nextPacket()
            return bodyPosY
    #################### Transitions ###########################

    def TD_event(self):
        t = self.robot.getTime()
        td1 = self.contact is False 
        td2 = (t-self.t_last_ev > self.tmin) 
        td3 = any(cnt > self.cnt_thr for cnt in self.contacts[-9:-1]) 
        td4 = (t-self.t_last_ev > self.tmax)
        #print("td_event: ", td)
        td = td1 and td2 and (td3 or td4)
        if td:
            #print("swimming phase lasted: " + str(t-self.t_last_ev))
            self.t_last_ev = t
            self.contact = True
        return td

    def LO_event(self):
        t = self.robot.getTime()
        lo = self.contact is True and (t-self.t_last_ev > self.tmin) and (all(cnt < self.deCnt_thr for cnt in self.contacts) or t-self.t_last_ev > self.tmax)
        if lo:
            #print("punting phase lasted: " + str(t-self.t_last_ev))
            self.contact = False            
            self.t_last_ev = t

        return lo    

    def OnTheGround_event(self):
        """to be implemented: the robot must subscribe to the status 
        publisher and read the y coordinate"""
        try:
            fail = self.headHitGround
        except:
            #print("Raising Exception")
            fail = False
        return fail

    def flight_state_transitions(self):
        self.motor[0].setPosition(self.q0_i)
        self.motor[0].setAvailableTorque(4)
        self.motor[1].setPosition(self.q1_i)
        self.motor[1].setAvailableTorque(4)
        if self.TD_event():
            newState = "Stance_state"
        elif self.OnTheGround_event():
            newState = "Failed_state"
        else:
            newState = "Flight_state"
        self.robot.step(self.timestep)
        return (newState)

    def stance_state_transitions(self):
        pos = np.maximum(0.01, self.q1_i-np.deg2rad(self.omega)*(self.robot.getTime()-self.t_last_ev))
        self.motor[0].setTorque(0.0)
        self.motor[1].setPosition(pos)
        self.motor[1].setAvailableTorque(4)

        if self.LO_event():
            newState = "Flight_state"
        elif self.OnTheGround_event():
            newState = "Failed_state"
        else:
            newState = "Stance_state"
        self.robot.step(self.timestep)
        return (newState)

    ##################  Run   #######################

    def run(self):

        filename = "sensorSet.csv"						# 7- Load the information on sensors from a csv file
        csvFile = pd.read_csv("/home/anna/Documenti/webots_projects/SingleLegLearner/controllers/supervisorManager/"+filename, sep=';')
        self.touch_no =int(len(csvFile.index))
        self.contacts = []
        self.TouchSensors = []
        names =[]
        for i in range(self.touch_no-1):
            names.append("TouchSensor_no"+str(i))
        
        for i in range(self.touch_no-3):
            self.TouchSensors.append(self.robot.getDevice(names[i]))
            self.contacts.append(0.0)
            try:
                self.TouchSensors[i].enable(self.timestep)
                #print("enabling sensor "+str(i))
            except:
                self.TouchSensors.remove(i)
                self.contacts.remove(i)
        """
        while self.robot.step(self.timestep) != 1:
            self.handle_receiver()
            self.handle_emitter() #turn off for open loop behaviors
        """    
# Create the robot controller object and run it
#robot_controller = SLL_RobotController()
#robot_controller.run()  # Run method is implemented by the framework, just need to call it
