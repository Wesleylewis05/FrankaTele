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

    POSE_ACTIONS = ["s", "w", "a", "d", "e", "q"]
    GRIP_ACTIONS = ["z"]
    ROT_ACTIONS = ["i", "k", "j", "l", "u", "o"]

    # Only these actions are exposed to gym environment.
    ACTIONS = POSE_ACTIONS + GRIP_ACTIONS + ROT_ACTIONS
    INIT_POS_DELTA = 0.01 # MAX_POS_DELTA = 0.1
    INIT_ROT_DELTA = 0.13  # Radian. MAX_ROT_DELTA = 0.2 

    def __init__(self,default_pos, default_ori):
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
        if k == "w":
            self.pos[0][0] -= self.pos_delta
        elif k == "s":
            self.pos[0][0] += self.pos_delta
        elif k == "a":
            self.pos[0][1] -= self.pos_delta
        elif k == "d":
            self.pos[0][1] += self.pos_delta
        elif k == "q":
            self.pos[0][2] -= self.pos_delta
        elif k == "e":
            self.pos[0][2] += self.pos_delta

    def _grip_action(self, k):
        if k == "z" and self.grasping == True:
            self.grasp = [30]
            self.grasping = False
        elif k == "z" and self.grasping == False:
            self.grasp = [0]
            self.grasping = True

    def _rot_action(self, k):
        if k == "k":
            self.ori[0][1] += self.rot_delta
        elif k == "i":
            self.ori[0][1] -= self.rot_delta
        elif k == "j":
            self.ori[0][0] += self.rot_delta
        elif k == "l":
            self.ori[0][0] -= self.rot_delta
        elif k == "o":
            self.ori[0][2] -= self.rot_delta
        elif k == "u":
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
        print("q (- z-axis) w (- x-axis) e (+ z-axis)")
        print("a (- y-axis) s (+ x-axis) d (+ y-axis)")

        print("Rotational movements in base frame")
        print("u (- z-axis-rot) i (neg y-axis-rot)  o (+ z-axis)")
        print("j (pos x-axis-rot) k (pos y-axis-rot)  l (neg x-axis-rot)")

        print("Toggle gripper open and close")
        print("z")
        print("===============================")

    def close(self):
        self.listener.stop()
