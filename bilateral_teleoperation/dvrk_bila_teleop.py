#!/usr/bin/env python

# Author: Junxiang Wang
# Date: 2024-04-12

# (C) Copyright 2024-2025 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

""" Bilateral teleoperation with neural network force estimation - ROS2 version """

import argparse
import crtk
from enum import Enum
import geometry_msgs.msg
import math
import numpy
import PyKDL
import std_msgs.msg
import sys
import time
import csv
import os
import cisstVectorPython as cisstVector
from scipy.spatial.transform import Rotation as R

try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    print("Warning: onnxruntime not available. Neural network force estimation will not work.")
    ONNXRUNTIME_AVAILABLE = False

DATA_PATH = "/home/hzhao78/cis2_bilateral_teleop/bilateral-teleop/model/dvrk_teleop_data/04_22_2026/"

class teleoperation:
    class State(Enum):
        ALIGNING = 1
        CLUTCHED = 2
        FOLLOWING = 3

    class LoadModel:
        def __init__(self, onnx_path, param_path):
            if not ONNXRUNTIME_AVAILABLE:
                raise ImportError("onnxruntime not available")
            try:
                self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                norm_data = numpy.load(param_path)
                self.input_mean = norm_data['input_mean']
                self.input_std = norm_data['input_std']
                self.target_mean = norm_data['target_mean']
                self.target_std = norm_data['target_std']
                
                # Try to infer seq_len from ONNX model input shape, default to 10 if not found
                try:
                    # Get input shape from ONNX model (typically [batch, seq_len, features])
                    input_shape = self.ort_session.get_inputs()[0].shape
                    if len(input_shape) > 1 and input_shape[1] is not None:
                        self.seq_len = input_shape[1]
                    else:
                        self.seq_len = 10  # default fallback
                except Exception:
                    self.seq_len = 10  # default fallback if inference fails
            except Exception as e:
                print(f"Error loading model {onnx_path} or params {param_path}: {e}")
                raise

    def __init__(self, ral, master, puppet, clutch_topic, run_period, align_mtm, operator_present_topic="", mode=""):
        if mode == "Neural Network":
            print('Initializing dvrk_teleoperation for {} and {} with neural network force estimation'.format(master.name, puppet.name))
        elif mode == "Data Collection":
            print('Initializing dvrk_teleoperation for {} and {} in data collection mode'.format(master.name, puppet.name))
        else:
            print('Initializing dvrk_teleoperation for {} and {}'.format(master.name, puppet.name))
        self.ral = ral
        self.run_period = run_period

        self.master = master
        self.puppet = puppet

        self.scale = 0.2
        self.velocity_scale = 0.2

        self.gripper_max = 60 * math.pi / 180
        self.gripper_zero = 0.0 # Set to e.g. 20 degrees if gripper cannot close past zero
        self.jaw_min = -20 * math.pi / 180
        self.jaw_max = 80 * math.pi / 180
        self.jaw_rate = 2 * math.pi

        self.can_align_mtm = align_mtm

        # slowly eliminate alignment offset if we can align mtm,
        # otherwise maintain fixed initial alignment offset
        self.align_rate = 0.25 * math.pi if self.can_align_mtm else 0.0

        # don't require alignment before beginning teleop if mtm wrist can't be actuated
        self.operator_orientation_tolerance = 5 * math.pi / 180 if self.can_align_mtm else math.pi
        self.operator_gripper_threshold = 5 * math.pi / 180
        self.operator_roll_threshold = 3 * math.pi / 180

        self.gripper_to_jaw_scale = self.jaw_max / (self.gripper_max - self.gripper_zero)
        self.gripper_to_jaw_offset = -self.gripper_zero * self.gripper_to_jaw_scale

        self.operator_is_active = False
        if operator_present_topic:
            self.operator_is_present = False
            self.operator_button = crtk.joystick_button(ral, operator_present_topic)
            self.operator_button.set_callback(self.on_operator_present)
        else:
            self.operator_is_present = True # if not given, then always assume present

        self.clutch_pressed = False
        self.clutch_button = crtk.joystick_button(ral, clutch_topic)
        self.clutch_button.set_callback(self.on_clutch)

        if mode == "Neural Network":
            # Neural network setup for force estimation - will be initialized when models are loaded
            self.seq_len = 10  # default, will be updated when models are loaded
            self.queue_master_first3 = None
            self.queue_master_last3 = None
            self.queue_puppet_first3 = None
            self.queue_puppet_last3 = None

        # control law gain
        self.force_gain = 0.2
        self.velocity_gain = 1.1

        if mode == "Data Collection":
            """for recording"""
            self.count = 0
            self.time_data = []
            self.y_data = []
            self.y_data_expected = []
            self.master_force = []
            self.puppet_force = []

            self.start_time = time.monotonic()
            self.recording_enabled = False
            self.record_size = 0

            self.output_csv_path = DATA_PATH + f"{self.start_time:.6f}-dVRK-Last-Train-joint_data.csv"
            self.csv_file = open(self.output_csv_path, "a", newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.header_written = os.path.getsize(self.output_csv_path) > 0

    def set_velocity_goal(self, v, base=1.10, max_gain=1.27, threshold=0.25):
        norm = numpy.linalg.norm(v)
        # print(f"v: {norm}")
        if norm < threshold:
            gain = max_gain  
        else:
            gain = max_gain * numpy.exp(-12 * ( norm - threshold))
            gain = numpy.maximum(base, gain)   
        return v * gain

    # def set_vctFrm3(self, rotation=None, translation=None):
    #     vctFrm3 = cisstVector.vctFrm3()
    #     if rotation is not None:
    #         assert all([isinstance(rotation, numpy.ndarray), rotation.shape == (3,3)])
    #         vctFrm3.SetRotation(rotation)
        
    #     if translation is not None:
    #         assert all([isinstance(translation, numpy.ndarray), translation.shape == (3,)])  #??
    #         vctFrm3.SetTranslation(translation)
    #     return vctFrm3
    
    # def GetRotAngle(self, Rot):
    #     r = R.from_matrix(Rot)
    #     rotvec = r.as_rotvec()

    #     theta = numpy.linalg.norm(rotvec)
    #     axis = rotvec / theta
    #     return theta, axis


    # def GetRotMatrix(self, axis, theta):
    #     # Ensure the axis is a unit vector
    #     axis = axis / numpy.linalg.norm(axis)

    #     rotvec = axis * theta
    #     r = R.from_rotvec(rotvec)
    #     rot_mat = r.as_matrix()
    #     return rot_mat
    
    # # average rotation by quaternion
    # def average_rotation(self, rotation1, alpha=1.0):
    #     # transfrom into scipy.rotation type
    #     rot_mats = numpy.array([rotation1])
    #     rots = R.from_matrix(rot_mats)

    #     weights = [alpha]
    #     mean_rot = rots.mean(weights=weights)
    #     mean_rot_mat = mean_rot.as_matrix()
    #     return mean_rot_mat

    # callback for operator pedal/button
    def on_operator_present(self, present):
        self.operator_is_present = present
        if not present:
            self.operator_is_active = False

    # callback for clutch pedal/button
    def on_clutch(self, clutch_pressed):
        self.clutch_pressed = clutch_pressed

    # compute relative orientation of mtm and psm
    def alignment_offset(self):
        return self.master.measured_cp()[0].M.Inverse() * self.puppet.setpoint_cp()[0].M

    # set relative origins for clutching and alignment offset
    def update_initial_state(self):
        self.master_cartesian_initial = self.master.measured_cp()[0]
        self.puppet_cartesian_initial = self.puppet.setpoint_cp()[0]
        self.alignment_offset_initial = self.alignment_offset()
        self.offset_angle, self.offset_axis = self.alignment_offset_initial.GetRotAngle()

    def gripper_to_jaw(self, gripper_angle):
        jaw_angle = self.gripper_to_jaw_scale * gripper_angle + self.gripper_to_jaw_offset

        # make sure we don't set goal past joint limits
        return max(jaw_angle, self.jaw_min)

    def jaw_to_gripper(self, jaw_angle):
        return (jaw_angle - self.gripper_to_jaw_offset) / self.gripper_to_jaw_scale

    def check_arm_state(self):
        if not self.puppet.is_homed():
            print(f'ERROR: {self.ral.node_name()}: puppet ({self.puppet.name}) is not homed anymore')
            self.running = False
        if not self.master.is_homed():
            print(f'ERROR: {self.ral.node_name()}: master ({self.master.name}) is not homed anymore')
            self.running = False

    # set relative origins for clutching and alignment offset
    def update_initial_state(self):
        self.master_cartesian_initial = self.master.measured_cp()[0]
        self.puppet_cartesian_initial = self.puppet.setpoint_cp()[0]
        self.alignment_offset_initial = self.alignment_offset()
        self.offset_angle, self.offset_axis = self.alignment_offset_initial.GetRotAngle()

    def gripper_to_jaw(self, gripper_angle):
        jaw_angle = self.gripper_to_jaw_scale * gripper_angle + self.gripper_to_jaw_offset
        # make sure we don't set goal past joint limits
        return max(jaw_angle, self.jaw_min)

    def jaw_to_gripper(self, jaw_angle):
        return (jaw_angle - self.gripper_to_jaw_offset) / self.gripper_to_jaw_scale

    def enter_aligning(self):
        self.current_state = teleoperation.State.ALIGNING
        self.last_align = None
        self.last_operator_prompt = time.perf_counter()

        self.master.use_gravity_compensation(True)
        self.puppet.hold()

        # reset operator activity data in case operator is inactive
        self.operator_roll_min = math.pi * 100
        self.operator_roll_max = -math.pi * 100
        self.operator_gripper_min = math.pi * 100
        self.operator_gripper_max = -math.pi * 100

    def transition_aligning(self):
        if self.operator_is_active and self.clutch_pressed:
            self.enter_clutched()
            return

        orientation_error, _ = self.alignment_offset().GetRotAngle()
        aligned = orientation_error <= self.operator_orientation_tolerance
        if aligned and self.operator_is_active:
            self.enter_following()

    def run_aligning(self):
        orientation_error, _ = self.alignment_offset().GetRotAngle()

        # if operator is inactive, use gripper or roll activity to detect when the user is ready
        if self.operator_is_present:
            gripper = self.master.gripper.measured_js()[0][0]
            self.operator_gripper_max = max(gripper, self.operator_gripper_max)
            self.operator_gripper_min = min(gripper, self.operator_gripper_min)
            gripper_range = self.operator_gripper_max - self.operator_gripper_min
            if gripper_range >= self.operator_gripper_threshold:
                self.operator_is_active = True

            # determine amount of roll around z axis by rotation of y-axis
            master_rotation, puppet_rotation = self.master.measured_cp()[0].M, self.puppet.setpoint_cp()[0].M
            master_y_axis = PyKDL.Vector(master_rotation[0,1], master_rotation[1,1], master_rotation[2,1])
            puppet_y_axis = PyKDL.Vector(puppet_rotation[0,1], puppet_rotation[1,1], puppet_rotation[2,1])
            roll = math.acos(PyKDL.dot(puppet_y_axis, master_y_axis))

            self.operator_roll_max = max(roll, self.operator_roll_max)
            self.operator_roll_min = min(roll, self.operator_roll_min)
            roll_range = self.operator_roll_max - self.operator_roll_min
            if roll_range >= self.operator_roll_threshold:
                self.operator_is_active = True

        # periodically send move_cp to MTM to align with PSM
        aligned = orientation_error <= self.operator_orientation_tolerance
        now = time.perf_counter()
        if not self.last_align or now - self.last_align > 4.0:
            move_cp = PyKDL.Frame(self.puppet.setpoint_cp()[0].M, self.master.setpoint_cp()[0].p)
            self.master.move_cp(move_cp)
            self.last_align = now

        # periodically notify operator if un-aligned or operator is inactive
        if self.operator_is_present and now - self.last_operator_prompt > 4.0:
            self.last_operator_prompt = now
            if not aligned:
                print(f'Unable to align master ({self.master.name}), angle error is {orientation_error * 180 / math.pi} (deg)')
            elif not self.operator_is_active:
                print(f'To begin teleop, pinch/twist master ({self.master.name}) gripper a bit')

    def enter_clutched(self):
        self.current_state = teleoperation.State.CLUTCHED

        # let MTM position move freely, but lock orientation
        wrench = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.master.body.servo_cf(wrench)
        self.master.lock_orientation(self.master.measured_cp()[0].M)

        self.puppet.hold()

    def transition_clutched(self):
        if not self.clutch_pressed or not self.operator_is_present:
            self.enter_aligning()

    def run_clutched(self):
        # let arm move freely
        pass

    def enter_following(self):
        self.current_state = teleoperation.State.FOLLOWING
        # update MTM/PSM origins position
        self.update_initial_state()

        # set up gripper ghost to rate-limit jaw speed
        jaw_setpoint = self.puppet.jaw.setpoint_js()[0]
        if len(jaw_setpoint) != 1:
            print(f'{self.ral.node_name()}: unable to get jaw position. Make sure there is an instrument on the puppet ({self.puppet.name})')
            self.running = False
        self.gripper_ghost = self.jaw_to_gripper(jaw_setpoint[0])

        self.master.use_gravity_compensation(True)

    def transition_following(self):
        if not self.operator_is_present:
            self.enter_aligning()
        elif self.clutch_pressed:
            self.enter_clutched()

    # Neural network force estimation
    def externalforce_prediction(self, component):
        # Check if models are loaded
        if not hasattr(component, 'firstmodel') or not hasattr(component, 'lastmodel'):
            print(f"Warning: Neural network models not loaded for {component.name}, using zero force")
            return numpy.zeros(6)
        
        # Initialize queues if not done yet
        if self.queue_master_first3 is None:
            self.seq_len = component.firstmodel.seq_len
            self.queue_master_first3 = numpy.zeros((1, self.seq_len, 6))
            self.queue_master_last3 = numpy.zeros((1, self.seq_len, 6))
            self.queue_puppet_first3 = numpy.zeros((1, self.seq_len, 6))
            self.queue_puppet_last3 = numpy.zeros((1, self.seq_len, 6))
        
        
        # measured_js returns joint positions, velocities, external forcees, and torques
        if component == self.master:
            measured_q, measured_dq, measured_torque = self.master.measured_js()
        else:
            measured_q, measured_dq, measured_torque = self.puppet.measured_js()    

        # Use first 6 joints (PSM has 6, MTM has 7)
        q = measured_q[:6]
        dq = measured_dq[:6]
        total_torque = measured_torque[:6]

        # normalize input
        first_input = numpy.concatenate((q[0:3], dq[0:3]))
        last_input = numpy.concatenate((q[3:6], dq[3:6]))

        first_input = (first_input - component.firstmodel.input_mean) / component.firstmodel.input_std
        last_input = (last_input - component.lastmodel.input_mean) / component.lastmodel.input_std

        first_input = numpy.expand_dims(first_input.reshape(1,-1), axis=0)    # shape(1,1,6)
        last_input = numpy.expand_dims(last_input.reshape(1,-1), axis=0)    # shape(1,1,6)

        if component == self.master:
            self.queue_master_first3 = numpy.concatenate((self.queue_master_first3, first_input), axis=1)
            self.queue_master_last3 = numpy.concatenate((self.queue_master_last3, last_input), axis = 1)
            self.queue_master_first3 = self.queue_master_first3[:, 1:, :]
            self.queue_master_last3 = self.queue_master_last3[:, 1:, :]

            # model predict
            first_ort_inputs = {component.firstmodel.ort_session.get_inputs()[0].name: self.queue_master_first3.astype(numpy.float32)}
            first_ort_outs = component.firstmodel.ort_session.run(None, first_ort_inputs)
            last_ort_inputs = {component.lastmodel.ort_session.get_inputs()[0].name: self.queue_master_last3.astype(numpy.float32)}
            last_ort_outs = component.lastmodel.ort_session.run(None, last_ort_inputs)

        else:  # puppet
            self.queue_puppet_first3 = numpy.concatenate((self.queue_puppet_first3, first_input), axis=1)
            self.queue_puppet_last3 = numpy.concatenate((self.queue_puppet_last3, last_input), axis = 1)
            self.queue_puppet_first3 = self.queue_puppet_first3[:, 1:, :]
            self.queue_puppet_last3 = self.queue_puppet_last3[:, 1:, :]

            # model predict
            first_ort_inputs = {component.firstmodel.ort_session.get_inputs()[0].name: self.queue_puppet_first3.astype(numpy.float32)}
            first_ort_outs = component.firstmodel.ort_session.run(None, first_ort_inputs)
            last_ort_inputs = {component.lastmodel.ort_session.get_inputs()[0].name: self.queue_puppet_last3.astype(numpy.float32)}
            last_ort_outs = component.lastmodel.ort_session.run(None, last_ort_inputs)

        # denormalize output
        torque_Joint1_3 = first_ort_outs[0]
        torque_Joint4_6 = last_ort_outs[0]

        torque_Joint1_3 = torque_Joint1_3 * component.firstmodel.target_std + component.firstmodel.target_mean
        torque_Joint4_6 = torque_Joint4_6 * component.lastmodel.target_std + component.lastmodel.target_mean

        internal_torque = numpy.hstack((torque_Joint1_3, torque_Joint4_6))
        external_torque = (total_torque - internal_torque)

        # convert to cartesian force
        if component.name.startswith('MTM'):
            external_torque = numpy.concatenate((external_torque, numpy.array([[measured_torque[6]]])), axis=1)
            internal_torque = numpy.concatenate((internal_torque, numpy.array([[measured_torque[6]]])), axis=1)

        J = component.body_jacobian()
        external_force = numpy.linalg.pinv(J.T) @ external_torque.T

        return external_force

    def run_following(self, mode=""):
        """
        Forward Process
        """
        # Force measurement using neural network
        if mode == "Neural Network":
            master_measured_cf = self.externalforce_prediction(self.master)
        else:
            master_measured_cf = self.master.body.measured_cf()[0]
        master_measured_cf[0:3] *= -1.0
        master_measured_cf[3:6] *= 0   # turn off torque

        if mode == "Neural Network":
            puppet_measured_cf = self.externalforce_prediction(self.puppet)
        else:
            puppet_measured_cf = self.puppet.body.measured_cf()[0]
        puppet_measured_cf[0:3] *= -1.0
        puppet_measured_cf[3:6] *= 0

        # force input
        if mode == "Neural Network":
            force_goal = self.force_gain * (master_measured_cf + puppet_measured_cf)
            force_goal = force_goal.reshape(-1).tolist()
            print(f"Neural network estimated force_goal: {force_goal}")
        else:
            alpha = 1.0
            gamma = 0.714
            force_goal = self.force_gain * (alpha * master_measured_cf + gamma * puppet_measured_cf)
            force_goal = force_goal.tolist()
            print(f"Measured force_goal: {force_goal}")

        # Position measurement
        master_measured_cp = self.master.measured_cp()[0]

        # set translation of psm
        master_translation = master_measured_cp.p - self.master_cartesian_initial.p
        master_translation *= self.scale
        puppet_position = master_translation + self.puppet_cartesian_initial.p

        # set rotation of psm to match mtm plus alignment offset
        max_delta = self.align_rate * self.run_period
        self.offset_angle += math.copysign(min(abs(self.offset_angle), max_delta), -self.offset_angle)

        alignment_offset = PyKDL.Rotation.Rot(self.offset_axis, self.offset_angle)
        puppet_rotation = master_measured_cp.M * alignment_offset

        # set cartesian goal of psm
        puppet_cartesian_goal = PyKDL.Frame(puppet_rotation, puppet_position)

        # Velocity measurement
        master_measured_cv = self.master.measured_cv()[0]
        master_measured_cv[0:3] *= self.velocity_scale
        master_measured_cv[3:6] *= 0.2
        puppet_velocity_goal = master_measured_cv.tolist()
        print(f"puppet_velocity_goal: {puppet_velocity_goal}")

        # Move
        self.puppet.servo_cp(puppet_cartesian_goal)
        self.puppet.servo_cv(puppet_velocity_goal)
        self.puppet.servo_cf(force_goal)

        ### Jaw/gripper teleop
        gripper_measured_js = self.master.gripper.measured_js()
        current_gripper = gripper_measured_js[0][0]

        ghost_lag = current_gripper - self.gripper_ghost
        max_delta = self.jaw_rate * self.run_period
        self.gripper_ghost += math.copysign(min(abs(ghost_lag), max_delta), ghost_lag)
        self.puppet.jaw.servo_jp(numpy.array([self.gripper_to_jaw(self.gripper_ghost)]))

        """
        Backward Process
        """
        # Position measurement
        puppet_measured_cp = self.puppet.measured_cp()[0]

        # set translation of mtm
        puppet_translation = puppet_measured_cp.p - self.puppet_cartesian_initial.p
        puppet_translation /= self.scale
        master_position = puppet_translation + self.master_cartesian_initial.p

        # set rotation of mtm
        master_rotation = puppet_measured_cp.M * alignment_offset.Inverse()

        # set cartesian goal of mtm
        master_cartesian_goal = PyKDL.Frame(master_rotation, master_position)

        # Velocity measurement
        puppet_measured_cv = self.puppet.measured_cv()[0]
        puppet_measured_cv[0:3] /= self.velocity_scale
        puppet_measured_cv[3:6] *= 0.2
        master_velocity_goal = puppet_measured_cv.tolist()

        # Move
        self.master.servo_cp(master_cartesian_goal)
        self.master.servo_cv(master_velocity_goal)
        self.master.servo_cf(force_goal)

        if mode == "Data Collection":
            """
            record plotting data
            """
            puppet_measured_cp_copy = PyKDL.Frame(puppet_measured_cp.M, puppet_measured_cp.p)
            self.y_data.append(puppet_measured_cp_copy)
            puppet_position_copy = PyKDL.Frame(PyKDL.Rotation.Identity(), puppet_position)
            self.y_data_expected.append(puppet_position_copy)
            # rotation = PyKDL.Rotation(
            #     master_measured_cf[0,0], master_measured_cf[0,1], master_measured_cf[0,2],
            #     master_measured_cf[1,0], master_measured_cf[1,1], master_measured_cf[1,2],
            #     master_measured_cf[2,0], master_measured_cf[2,1], master_measured_cf[2,2]
            # )
            # position = PyKDL.Vector(master_measured_cf[0,3], master_measured_cf[1,3], master_measured_cf[2,3])
            # master_measured_cf_copy = PyKDL.Frame(rotation, position)
            master_measured_cf_copy = master_measured_cf.copy()
            self.master_force.append(master_measured_cf_copy)
            # rotation = PyKDL.Rotation(
            #     puppet_measured_cf[0,0], puppet_measured_cf[0,1], puppet_measured_cf[0,2],
            #     puppet_measured_cf[1,0], puppet_measured_cf[1,1], puppet_measured_cf[1,2],
            #     puppet_measured_cf[2,0], puppet_measured_cf[2,1], puppet_measured_cf[2,2]
            # )
            # position = PyKDL.Vector(puppet_measured_cf[0,3], puppet_measured_cf[1,3], puppet_measured_cf[2,3])
            # puppet_measured_cf_copy = PyKDL.Frame(rotation, position)
            puppet_measured_cf_copy = puppet_measured_cf.copy()
            self.puppet_force.append(puppet_measured_cf_copy)


            '''For recording'''
            current_time = time.monotonic()
            print(f"recording enabled: {self.recording_enabled}")
            if not self.recording_enabled and float(current_time - self.start_time) >= 30.0:
                print("Start recording joint data")
                self.recording_enabled = True

            if self.recording_enabled and self.record_size >= 200000:
                print("Auto stopping: 200000 series of data acquired.")
                # time.strftime("%Y-%m-%d %H:%M:%S", current_time)
                # time.strftime("%Y-%m-%d %H:%M:%S", self.start_time)
                print(f"start_time: {self.start_time}")
                print(f"end_time: {current_time}")
                self.recording_enabled = False
                self.running = False

            if self.recording_enabled:
                self.record_size += 1
                print("Recording data.")

                timestamp = time.time()

                master_q, master_dq, master_torque = self.master.measured_js()
                master_q = master_q[:6].tolist()
                master_dq = master_dq[:6].tolist()
                master_torque = master_torque[:6].tolist()

                puppet_q, puppet_dq, puppet_torque = self.puppet.measured_js()
                puppet_q = puppet_q.tolist()
                puppet_dq = puppet_dq.tolist()
                puppet_torque = puppet_torque.tolist()

                row = [timestamp] + master_q + master_dq + master_torque + puppet_q + puppet_dq + puppet_torque

                if not self.header_written:
                    headers = ['timestamp'] + \
                            [f'master_q{i}' for i in range(6)] + [f'master_dq{i}' for i in range(6)] + [f'master_torque{i}' for i in range(6)] + \
                            [f'puppet_q{i}' for i in range(6)]  + [f'puppet_dq{i}' for i in range(6)]  + [f'puppet_torque{i}' for i in range(6)]
                    self.csv_writer.writerow(headers)
                    self.header_written = True

                self.csv_writer.writerow(row)

    def home(self):
        print("Homing arms...")
        timeout = 10.0 # seconds
        if not self.puppet.enable(timeout) or not self.puppet.home(timeout):
            print('    ! failed to home {} within {} seconds'.format(self.puppet.name, timeout))
            return False

        if not self.master.enable(timeout) or not self.master.home(timeout):
            print('    ! failed to home {} within {} seconds'.format(self.master.name, timeout))
            return False

        print("    Homing is complete")
        return True

    def run(self):
        homed_successfully = self.home()
        if not homed_successfully:
            print("home not success")
            return

        teleop_rate = self.ral.create_rate(int(1/self.run_period))
        print("Running teleop at {} Hz".format(int(1/self.run_period)))

        self.enter_aligning()
        self.running = True

        self.master.lock_orientation(self.master.measured_cp()[0].M)

        while not self.ral.is_shutdown():
            # check if teleop state should transition
            if self.current_state == teleoperation.State.ALIGNING:
                self.transition_aligning()
            elif self.current_state == teleoperation.State.CLUTCHED:
                self.transition_clutched()
            elif self.current_state == teleoperation.State.FOLLOWING:
                self.transition_following()
            else:
                raise RuntimeError("Invalid state: {}".format(self.current_state))

            self.check_arm_state()

            if not self.running:
                break

            # run teleop state handler
            if self.current_state == teleoperation.State.ALIGNING:
                self.run_aligning()
            elif self.current_state == teleoperation.State.CLUTCHED:
                self.run_clutched()
            elif self.current_state == teleoperation.State.FOLLOWING:
                self.run_following(mode=args.mode)
            else:
                raise RuntimeError("Invalid state: {}".format(self.current_state))

            teleop_rate.sleep()
        
        if args.mode == "Data Collection":
            # save data
            numpy.savetxt(DATA_PATH + 'multi_array.txt', self.y_data, fmt='%f', delimiter=' ', comments='')
            numpy.savetxt(DATA_PATH + 'multi_array_exp.txt', self.y_data_expected, fmt='%f', delimiter=' ', comments='')
            numpy.savetxt(DATA_PATH + 'PSML_total_force.txt', self.puppet_force, fmt='%f', delimiter=' ', comments='')
            numpy.savetxt(DATA_PATH + 'MTML_total_force.txt', self.master_force, fmt='%f', delimiter=' ', comments='')
            print(f"data.txt saved!")

        print(f"Program finished!")


class MTM:
    class ServoMeasCF:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_servo_cf()
            self.utils.add_measured_cf()
            self.utils.add_measured_js()
            self.utils.add_jacobian()

    class Gripper:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_measured_js()

    class MeasuredJS:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_measured_js()

    def __init__(self, ral, arm_name, timeout, firstjoints_onnxpath=None, firstjoints_parampath=None, lastjoints_onnxpath=None, lastjoints_parampath=None, mode=""):
        self.name = arm_name
        self.ral = ral.create_child(arm_name)
        self.utils = crtk.utils(self, self.ral, timeout)

        self.utils.add_operating_state()
        self.utils.add_measured_cp()
        self.utils.add_measured_cv()
        self.utils.add_measured_cf()
        self.utils.add_setpoint_cp()
        self.utils.add_move_cp()
        self.utils.add_servo_cf()
        self.utils.add_servo_cp()
        self.utils.add_servo_cv()
        self.utils.add_servo_jf()
        self.utils.add_jacobian()

        self.gripper = self.Gripper(self.ral.create_child('gripper'), timeout)
        self.measured_root = self.MeasuredJS(self.ral, timeout)
        self.body = self.ServoMeasCF(self.ral.create_child('body'), timeout)

        if mode == "Neural Network":
            # Load neural network models
            if ONNXRUNTIME_AVAILABLE and firstjoints_onnxpath is not None and firstjoints_parampath is not None:
                try:
                    self.firstmodel = teleoperation.LoadModel(firstjoints_onnxpath, firstjoints_parampath)
                except Exception as e:
                    print(f"Failed to load first model for {arm_name}: {e}")
            if ONNXRUNTIME_AVAILABLE and lastjoints_onnxpath is not None and lastjoints_parampath is not None:
                try:
                    self.lastmodel = teleoperation.LoadModel(lastjoints_onnxpath, lastjoints_parampath)
                except Exception as e:
                    print(f"Failed to load last model for {arm_name}: {e}")

        # non-CRTK topics
        self.lock_orientation_pub = self.ral.publisher('lock_orientation',
                                                        geometry_msgs.msg.Quaternion,
                                                        latch = True, queue_size = 1)
        self.unlock_orientation_pub = self.ral.publisher('unlock_orientation',
                                                         std_msgs.msg.Empty,
                                                         latch = True, queue_size = 1)
        self.use_gravity_compensation_pub = self.ral.publisher('use_gravity_compensation',
                                                                std_msgs.msg.Bool,
                                                                latch = True, queue_size = 1)
        
    def lock_orientation(self, orientation):
        """orientation should be a PyKDL.Rotation object"""
        q = geometry_msgs.msg.Quaternion()
        q.x, q.y, q.z, q.w = orientation.GetQuaternion()
        self.lock_orientation_pub.publish(q)

    def unlock_orientation(self):
        self.unlock_orientation_pub.publish(std_msgs.msg.Empty())

    def use_gravity_compensation(self, gravity_compensation):
        """Turn on/off gravity compensation (only applies to Cartesian effort mode)"""
        msg = std_msgs.msg.Bool(data=gravity_compensation)
        self.use_gravity_compensation_pub.publish(msg)

    def measured_js(self):
        try:
            measured_js = self.measured_root.measured_js()
        except AttributeError:
            try:
                measured_js = self.body.measured_js()
            except AttributeError:
                print(f"Warning: measured_js not available for {self.name}")
                raise
        return measured_js[0], measured_js[1], measured_js[2]  # position, velocity, effort

    def body_jacobian(self):
        # Get the Jacobian matrix for the body
        try:
            j, _ = self.body.jacobian()
            return j.copy()
        except AttributeError:
            # Fallback: return identity matrix
            print(f"Warning: Jacobian not available for {self.name}, using identity")
            return numpy.eye(6, 7)  # MTM has 7 joints


class PSM:
    class MeasureCF:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_measured_cf()
            self.utils.add_measured_js()
            self.utils.add_jacobian()

    class MeasuredJS:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_measured_js()

    class MeasuredCP:
        def __init__(self,ral,timeout):
            self.utils = crtk.utils(self,ral,timeout)
            self.utils.add_measured_cp()

    class Jaw:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_setpoint_js()
            self.utils.add_servo_jp()

    def __init__(self, ral, arm_name, timeout, firstjoints_onnxpath=None, firstjoints_parampath=None, lastjoints_onnxpath=None, lastjoints_parampath=None, mode=""):
        self.name = arm_name
        self.ral = ral.create_child(arm_name)
        self.utils = crtk.utils(self, self.ral, timeout)

        self.utils.add_operating_state()
        self.utils.add_setpoint_cp()
        self.utils.add_servo_cp()
        self.utils.add_servo_cf()
        self.utils.add_servo_cv() # added
        self.utils.add_hold()
        self.utils.add_measured_cv()
        self.utils.add_move_jp()
        self.utils.add_measured_cp()
        self.utils.add_jacobian()
        
        self.body = self.MeasureCF(self.ral.create_child('body'), timeout)
        self.measured_root = self.MeasuredJS(self.ral, timeout)
        self.local = self.MeasuredCP(self.ral.create_child('local'),timeout)
        self.jaw = self.Jaw(self.ral.create_child('jaw'), timeout)

        if mode == "Neural Network":
            # Load neural network models
            if ONNXRUNTIME_AVAILABLE and firstjoints_onnxpath is not None and firstjoints_parampath is not None:
                try:
                    self.firstmodel = teleoperation.LoadModel(firstjoints_onnxpath, firstjoints_parampath)
                except Exception as e:
                    print(f"Failed to load first model for {arm_name}: {e}")
            if ONNXRUNTIME_AVAILABLE and lastjoints_onnxpath is not None and lastjoints_parampath is not None:
                try:
                    self.lastmodel = teleoperation.LoadModel(lastjoints_onnxpath, lastjoints_parampath)
                except Exception as e:
                    print(f"Failed to load last model for {arm_name}: {e}")

    def measured_js(self):
        try:
            measured_js = self.measured_root.measured_js()
        except AttributeError:
            try:
                measured_js = self.body.measured_js()
            except AttributeError:
                print(f"Warning: measured_js not available for {self.name}")
                raise
        return measured_js[0], measured_js[1], measured_js[2]  # position, velocity, effort

    def body_jacobian(self):
        # Get the Jacobian matrix for the body
        try:
            j, _ = self.body.jacobian()
            return j.copy()
        except AttributeError:
            # Fallback: return identity matrix
            print(f"Warning: Jacobian not available for {self.name}, using identity")
            return numpy.eye(6, 6)  # PSM has 6 joints


if __name__ == '__main__':
    # extract ros arguments (e.g. __ns:= for namespace)
    argv = crtk.ral.parse_argv(sys.argv[1:]) # skip argv[0], script name

    # parse arguments
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mtm', type = str, required = True,
                        choices = ['MTML', 'MTMR'],
                        help = 'MTM arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-p', '--psm', type = str, required = True,
                        choices = ['PSM1', 'PSM2', 'PSM3'],
                        help = 'PSM arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-c', '--clutch', type = str, default='/console1/clutch',
                        help = 'ROS topic corresponding to clutch button/pedal input')
    parser.add_argument('-o', '--operator', type = str, default='/console1/operator_present', const=None, nargs='?',
                        help = 'ROS topic corresponding to operator present button/pedal/sensor input - use "-o" without an argument to disable')
    parser.add_argument('-n', '--no-mtm-alignment', action='store_true',
                        help="don't align mtm (useful for using haptic devices as MTM which don't have wrist actuation)")
    parser.add_argument('-i', '--interval', type=float, default=0.001,
                        help = 'time interval/period to run at - should be as long as console\'s period to prevent timeouts')
    parser.add_argument('-t', '--mode', type = str, required = True,
                        choices = ['Normal', 'Neural Network', 'Data Collection'],
                        help = 'Operation mode for the teleoperation')
    parser.add_argument('--model-path', type=str, default='/home/hzhao78/cis2_bilateral_teleop/multilateral-teleop/model/training_results',
                        help = 'Path to the neural network model files')
    args = parser.parse_args(argv)

    ral = crtk.ral('dvrk_python_teleoperation')

    if args.mode == 'Neural Network':
        # Initialize model path variables
        master_first_onnx = ""
        master_first_params = ""
        master_last_onnx = ""
        master_last_params = ""
        puppet_first_onnx = ""
        puppet_first_params = ""
        puppet_last_onnx = ""
        puppet_last_params = ""

        # Determine model paths based on MTM and PSM names
        # type = 'best' #'final'
        # if args.mtm == 'MTML':
        #     master_first_onnx = f"{args.model_path}/master-FirstThreeJoints_{type}.onnx"
        #     master_first_params = f"{args.model_path}/master-FirstThreeJoints_{type}_Parameters.npz"
        #     master_last_onnx = f"{args.model_path}/master-LastThreeJoints_{type}.onnx"
        #     master_last_params = f"{args.model_path}/master-LastThreeJoints_{type}_Parameters.npz"
        # else:  # MTMR
        #     master_first_onnx = f"{args.model_path}/master-FirstThreeJoints_{type}.onnx"
        #     master_first_params = f"{args.model_path}/master-FirstThreeJoints_{type}_Parameters.npz"
        #     master_last_onnx = f"{args.model_path}/master-LastThreeJoints_{type}.onnx"
        #     master_last_params = f"{args.model_path}/master-LastThreeJoints_{type}_Parameters.npz"

        # puppet_first_onnx = f"{args.model_path}/puppet-FirstThreeJoints_{type}.onnx"
        # puppet_first_params = f"{args.model_path}/puppet-FirstThreeJoints_{type}_Parameters.npz"
        # puppet_last_onnx = f"{args.model_path}/puppet-LastThreeJoints_{type}.onnx"
        # puppet_last_params = f"{args.model_path}/puppet-LastThreeJoints_{type}_Parameters.npz"

        if args.mtm == 'MTML':
            master_first_onnx = f"{args.model_path}/0628-Mul-master1-First.onnx"
            master_first_params = f"{args.model_path}/0628-master1-First-norm_params.npz"
            master_last_onnx = f"{args.model_path}/0628-Mul-master1-Last.onnx"
            master_last_params = f"{args.model_path}/0628-master1-Last-norm_params.npz"
        else:  # MTMR
            master_first_onnx = f"{args.model_path}/0628-Mul-master2-First.onnx"
            master_first_params = f"{args.model_path}/0628-Mul-master2-First-norm_params.npz"
            master_last_onnx = f"{args.model_path}/0628-Mul-master2-Last.onnx"
            master_last_params = f"{args.model_path}/0628-Mul-master2-Last-norm_params.npz"

        puppet_first_onnx = f"{args.model_path}/0628-Mul-puppet-First.onnx"
        puppet_first_params = f"{args.model_path}/0628-puppet-First-norm_params.npz"
        puppet_last_onnx = f"{args.model_path}/0628-Mul-puppet-Last.onnx"
        puppet_last_params = f"{args.model_path}/0628-puppet-Last-norm_params.npz"

        mtm = MTM(ral, args.mtm, timeout=20*args.interval,
                firstjoints_onnxpath=master_first_onnx,
                firstjoints_parampath=master_first_params,
                lastjoints_onnxpath=master_last_onnx,
                lastjoints_parampath=master_last_params,
                mode=args.mode)
        psm = PSM(ral, args.psm, timeout=20*args.interval,
                firstjoints_onnxpath=puppet_first_onnx,
                firstjoints_parampath=puppet_first_params,
                lastjoints_onnxpath=puppet_last_onnx,
                lastjoints_parampath=puppet_last_params,
                mode=args.mode)
    else:
        mtm = MTM(ral, args.mtm, timeout=20*args.interval, mode=args.mode)
        psm = PSM(ral, args.psm, timeout=20*args.interval, mode=args.mode)

    application = teleoperation(ral, mtm, psm, args.clutch, args.interval, not args.no_mtm_alignment, operator_present_topic=args.operator, mode=args.mode)
    ral.spin_and_execute(application.run)