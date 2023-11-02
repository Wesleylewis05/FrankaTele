"""Reference: https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/devices/keyboard.py"""

import gym
import numpy as np
from pynput.keyboard import Key, Listener
import torch
from device.device_interface import DeviceInterface
from device.collect_enum import CollectEnum
import device.transform as T


class KeyboardInterface(DeviceInterface):
    """Define keyboard interface to control franka."""

    POSE_ACTIONS = None
    GRIP_ACTIONS = None
    ROT_ACTIONS = None

    # Only these actions are exposed to gym environment.
    ACTIONS = None
    INIT_POS_DELTA = 0.01 # MAX_POS_DELTA = 0.1
    INIT_ROT_DELTA = 0.13  # Radian. MAX_ROT_DELTA = 0.2 

    def __init__(self,default_pos, default_ori, pose_actions=None, grip_actions=None, rot_actions=None):
        # Assign the action mappings if provided, otherwise use defaults
        KeyboardInterface.POSE_ACTIONS = pose_actions if pose_actions is not None else ["s", "w", "a", "d", "e", "q"]
        KeyboardInterface.GRIP_ACTIONS = grip_actions if grip_actions is not None else ["z"]
        KeyboardInterface.ROT_ACTIONS = rot_actions if rot_actions is not None else ["i", "k", "j", "l", "u", "o"]
        KeyboardInterface.ACTIONS = KeyboardInterface.POSE_ACTIONS + KeyboardInterface.GRIP_ACTIONS + KeyboardInterface.ROT_ACTIONS
        self.default_pos = default_pos
        self.default_ori = default_ori
        self.reset()
        self.speed_factor = 0.5
        self.pos_delta = KeyboardInterface.INIT_POS_DELTA * self.speed_factor
        self.rot_delta = KeyboardInterface.INIT_ROT_DELTA * self.speed_factor
        # Make a thread to listen to keyboard and register callback functions.
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
               

    def reset(self):
        self.pos = self.default_pos
        self.last_pos = torch.clone(self.pos)
        self.ori = self.default_ori
        self.last_ori = torch.clone(self.ori)
        self.grasp = [0]
        self.grasping = True 
        self.print_usage()

    def on_press(self, k):
        try:
            k = k.char

            # Moving arm.
            if k in KeyboardInterface.ACTIONS:
                if k in KeyboardInterface.POSE_ACTIONS:
                    self._pose_action(k)
                elif k in KeyboardInterface.GRIP_ACTIONS:
                    self._grip_action(k)
                elif k in KeyboardInterface.ROT_ACTIONS:
                    self._rot_action(k)

            # Data labelling and debugging.
            elif k in KeyboardInterface.ADJUST_DELTA:
                self._adjust_delta(k)
            elif k == "t":
                self.key_enum = CollectEnum.SUCCESS
            elif k == "n":
                self.key_enum = CollectEnum.FAIL
            elif k.isdigit():
                gym.logger.info(f"Reward pressed: {k}")
                self.rew_key = int(k)
                self.key_enum = CollectEnum.REWARD
            elif k == "`":
                gym.logger.info("Skill complete pressed")
                self.key_enum = CollectEnum.SKILL
            elif k == "r":
                gym.logger.info("Reset pressed")
                self.key_enum = CollectEnum.RESET
        except AttributeError as e:
            pass

    def on_release(self, k):
        try:
            # Terminates keyboard monitoring.
            if k == Key.esc:
                return False
        except AttributeError as e:
            pass

    def _pose_action(self, k):
        if k == KeyboardInterface.ACTIONS[1]:
            self.pos[0][0] -= self.pos_delta
        elif k == KeyboardInterface.ACTIONS[0]:
            self.pos[0][0] += self.pos_delta
        elif k == KeyboardInterface.ACTIONS[2]:
            self.pos[0][1] -= self.pos_delta
        elif k == KeyboardInterface.ACTIONS[3]:
            self.pos[0][1] += self.pos_delta
        elif k == KeyboardInterface.ACTIONS[5]:
            self.pos[0][2] -= self.pos_delta
        elif k == KeyboardInterface.ACTIONS[4]:
            self.pos[0][2] += self.pos_delta

    def _grip_action(self, k):
        if k == KeyboardInterface.ACTIONS[6] and self.grasping == True:
            self.grasp = [30]
            self.grasping = False
        elif k == KeyboardInterface.ACTIONS[6] and self.grasping == False:
            self.grasp = [0]
            self.grasping = True

    def _rot_action(self, k):
        if k == KeyboardInterface.ACTIONS[8]:
            self.ori[0][1] += self.rot_delta
        elif k == KeyboardInterface.ACTIONS[7]:
            self.ori[0][1] -= self.rot_delta
        elif k == KeyboardInterface.ACTIONS[9]:
            self.ori[0][0] += self.rot_delta
        elif k == KeyboardInterface.ACTIONS[10]:
            self.ori[0][0] -= self.rot_delta
        elif k == KeyboardInterface.ACTIONS[12]:
            self.ori[0][2] -= self.rot_delta
        elif k == KeyboardInterface.ACTIONS[11]:
            self.ori[0][2] += self.rot_delta

    def get_action(self):
        dpos = self.pos
        dori = self.ori
        self.last_pos = torch.clone(self.pos)
        self.last_ori = torch.clone(self.ori)
        return dpos, dori, self.grasp

    def print_usage(self):
        print("==============Keyboard Usage=================")
        print("Positional movements in base frame")
        print(f"{KeyboardInterface.ACTIONS[5]} (- z-axis) {KeyboardInterface.ACTIONS[1]} (- x-axis) {KeyboardInterface.ACTIONS[4]} (+ z-axis)")
        print(f"{KeyboardInterface.ACTIONS[2]} (- y-axis) {KeyboardInterface.ACTIONS[0]} (+ x-axis) {KeyboardInterface.ACTIONS[3]} (+ y-axis)")

        print("Rotational movements in base frame")
        print(f"{KeyboardInterface.ACTIONS[11]} (- z-axis-rot) {KeyboardInterface.ACTIONS[7]} (neg y-axis-rot)  {KeyboardInterface.ACTIONS[12]} (+ z-axis)")
        print(f"{KeyboardInterface.ACTIONS[9]} (pos x-axis-rot) {KeyboardInterface.ACTIONS[8]} (pos y-axis-rot)  {KeyboardInterface.ACTIONS[10]} (neg x-axis-rot)")

        print("Toggle gripper open and close")
        print(f"{KeyboardInterface.ACTIONS[6]}")
        print("===============================")

    def close(self):
        self.listener.stop()
