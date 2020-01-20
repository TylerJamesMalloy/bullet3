from .scene_abstract import SingleRobotEmptyScene
from .env_bases import MJCFBaseBulletEnv
from robot_pendula import InvertedPendulum, InvertedPendulumSwingup, InvertedDoublePendulum
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
import os, sys


class InvertedPendulumBulletEnv(MJCFBaseBulletEnv):

  def __init__(self):
    self.robot = InvertedPendulum()
    MJCFBaseBulletEnv.__init__(self, self.robot)
    self.stateId = -1

    self.torqueForce = 100
    self.gravity = 9.8
    self.poleMass = 5

  def create_single_player_scene(self, bullet_client):
    return SingleRobotEmptyScene(bullet_client, gravity=self.gravity, timestep=0.0165, frame_skip=1)

  def reset(self):
    if (self.stateId >= 0):
      #print("InvertedPendulumBulletEnv reset p.restoreState(",self.stateId,")")
      self._p.restoreState(self.stateId)
    r = MJCFBaseBulletEnv.reset(self)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
      #print("InvertedPendulumBulletEnv reset self.stateId=",self.stateId)
    return r

  def get_features(self):
    return self.torqueForce, self.gravity, self.poleMass
  
  def set_features(self, torqueForce=100, gravity=9.8, mass=5):
    self.reset()
    
    self.torqueForce = torqueForce
    self.gravity = gravity
    self.poleMass = mass

    self._p.setGravity(0, 0, -self.gravity)

    bodyIndex = self.robot.parts["pole"].bodyIndex
    bodyUniqueId = self.robot.parts["pole"].bodies[bodyIndex]
    partIndex = self.robot.parts["pole"].bodyPartIndex

    info = self._p.getDynamicsInfo(bodyUniqueId, partIndex)
    self._p.changeDynamics(bodyUniqueId, partIndex, mass = self.poleMass)

    info = self._p.getDynamicsInfo(bodyUniqueId, partIndex)


  def randomize(self, level=0):
    self.reset()

    if(level == 0):
      self.torqueForce = self.np_random.uniform(low=90, high=110)
      self.gravity = self.np_random.uniform(low=8.82, high=10.78)
      self.poleMass = self.np_random.uniform(low=4.5, high=5.5)

    elif(level == 1):
      if(self.np_random.uniform(low=0, high=0.5) > 0.5):
        self.torqueForce = self.np_random.uniform(low=75, high=90)
      else:
        self.torqueForce = self.np_random.uniform(low=110, high=125)
      
      if(self.np_random.uniform(low=0, high=0.5) > 0.5):
        self.gravity = self.np_random.uniform(low=7.35, high=8.82)
      else:
        self.gravity = self.np_random.uniform(low=10.78, high=12.25)

      if(self.np_random.uniform(low=0, high=0.5) > 0.5):
        self.poleMass = self.np_random.uniform(low=3.75, high=4.5)
      else:
        self.poleMass = self.np_random.uniform(low=5.5, high=6.25)

    else:
      print("ERROR: Invalid randomization mode selected, only 0 or 1 are available")
    
    #print("randomizing, torque: ", self.torqueForce  , " gravity ", self.gravity)

    self._p.setGravity(0, 0, -self.gravity)

    bodyIndex = self.robot.parts["pole"].bodyIndex
    bodyUniqueId = self.robot.parts["pole"].bodies[bodyIndex]
    partIndex = self.robot.parts["pole"].bodyPartIndex

    info = self._p.getDynamicsInfo(bodyUniqueId, partIndex)
    self._p.changeDynamics(bodyUniqueId, partIndex, mass = self.poleMass)

    info = self._p.getDynamicsInfo(bodyUniqueId, partIndex)
  
  def apply_action(self, a):
    assert (np.isfinite(a).all())
    if not np.isfinite(a).all():
      print("a is inf")
      a[0] = 0
    self.slider.set_motor_torque(self.torqueForce * float(np.clip(a[0], -1, +1)))

  def step(self, a):
    self.robot.apply_action(a)
    self.scene.global_step()
    state = self.robot.calc_state()  # sets self.pos_x self.pos_y
    vel_penalty = 0
    if self.robot.swingup:
      reward = np.cos(self.robot.theta)
      done = False
    else:
      reward = 1.0
      done = np.abs(self.robot.theta) > .2
    self.rewards = [float(reward)]
    self.HUD(state, a, done)
    return state, sum(self.rewards), done, {}

  def camera_adjust(self):
    self.camera.move_and_look_at(0, 1.2, 1.0, 0, 0, 0.5)


class InvertedPendulumSwingupBulletEnv(InvertedPendulumBulletEnv):

  def __init__(self):
    self.robot = InvertedPendulumSwingup()
    MJCFBaseBulletEnv.__init__(self, self.robot)
    self.stateId = -1


class InvertedDoublePendulumBulletEnv(MJCFBaseBulletEnv):

  def __init__(self):
    self.robot = InvertedDoublePendulum()
    MJCFBaseBulletEnv.__init__(self, self.robot)
    self.stateId = -1
  
  def randomize(self, level=0):
    print("randomizing")

  def create_single_player_scene(self, bullet_client):
    return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

  def reset(self):
    if (self.stateId >= 0):
      self._p.restoreState(self.stateId)
    r = MJCFBaseBulletEnv.reset(self)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
    return r

  def step(self, a):
    self.robot.apply_action(a)
    self.scene.global_step()
    state = self.robot.calc_state()  # sets self.pos_x self.pos_y
    # upright position: 0.6 (one pole) + 0.6 (second pole) * 0.5 (middle of second pole) = 0.9
    # using <site> tag in original xml, upright position is 0.6 + 0.6 = 1.2, difference +0.3
    dist_penalty = 0.01 * self.robot.pos_x**2 + (self.robot.pos_y + 0.3 - 2)**2
    # v1, v2 = self.model.data.qvel[1:3]   TODO when this fixed https://github.com/bulletphysics/bullet3/issues/1040
    #vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
    vel_penalty = 0
    alive_bonus = 10
    done = self.robot.pos_y + 0.3 <= 1
    self.rewards = [float(alive_bonus), float(-dist_penalty), float(-vel_penalty)]
    self.HUD(state, a, done)
    return state, sum(self.rewards), done, {}

  def camera_adjust(self):
    self.camera.move_and_look_at(0, 1.2, 1.2, 0, 0, 0.5)
