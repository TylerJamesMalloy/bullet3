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
      self._p.restoreState(self.stateId)
    r = MJCFBaseBulletEnv.reset(self)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()


    # Reset dynamics after environment reset

    self._p.setGravity(0, 0, -self.gravity)

    bodyIndex = self.robot.parts["pole"].bodyIndex
    bodyUniqueId = self.robot.parts["pole"].bodies[bodyIndex]
    partIndex = self.robot.parts["pole"].bodyPartIndex
    
    self._p.changeDynamics(bodyUniqueId, partIndex, mass = self.poleMass)

    return r

  def get_features(self):
    return self.torqueForce, self.gravity, self.poleMass
  
  def set_features(self, features = [100, 9.8, 5]):
    self.reset()

    torqueForce = features[0]
    gravity = features[1]
    mass = features[2]
    
    self.torqueForce = torqueForce
    self.gravity = gravity
    self.poleMass = mass

    self._p.setGravity(0, 0, -self.gravity)

    bodyIndex = self.robot.parts["pole"].bodyIndex
    bodyUniqueId = self.robot.parts["pole"].bodies[bodyIndex]
    partIndex = self.robot.parts["pole"].bodyPartIndex
    self._p.changeDynamics(bodyUniqueId, partIndex, mass = self.poleMass)

  def randomize(self, level=0):
    self.reset()

    if(level == 0):
      self.torqueForce = self.np_random.uniform(low=90, high=110)
      self.gravity = self.np_random.uniform(low=8.82, high=10.78)
      self.poleMass = self.np_random.uniform(low=4.5, high=5.5)

    elif(level == 1):
      if(self.np_random.uniform(low=0, high=1.0) > 0.5):
        self.torqueForce = self.np_random.uniform(low=50, high=75)
      else:
        self.torqueForce = self.np_random.uniform(low=125, high=150)
      
      if(self.np_random.uniform(low=0, high=1.0) > 0.5):
        self.gravity = self.np_random.uniform(low=4.9, high=7.35)
      else:
        self.gravity = self.np_random.uniform(low=12.25, high=14.7)

      if(self.np_random.uniform(low=0, high=1.0) > 0.5):
        self.poleMass = self.np_random.uniform(low=2.5, high=3.75)
      else:
        self.poleMass = self.np_random.uniform(low=6.25, high=7.5)

    else:
      print("ERROR: Invalid randomization mode selected, only 0 or 1 are available")
    
    #print("randomizing, torque: ", self.torqueForce  , " gravity ", self.gravity)

    self._p.setGravity(0, 0, -self.gravity)

    bodyIndex = self.robot.parts["pole"].bodyIndex
    bodyUniqueId = self.robot.parts["pole"].bodies[bodyIndex]
    partIndex = self.robot.parts["pole"].bodyPartIndex

    self._p.changeDynamics(bodyUniqueId, partIndex, mass = self.poleMass)
  
  def apply_action(self, a):
    assert (np.isfinite(a).all())
    if not np.isfinite(a).all():
      print("a is inf")
      a[0] = 0

    self.robot.slider.set_motor_torque(self.torqueForce * float(np.clip(a[0], -1, +1)))

  def step(self, a):
    #self.robot.apply_action(a) # Need to have the torqueForce parameter to apply the correct action
    self.apply_action(a)
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

    self.torqueForce = 100
    self.gravity = 9.8
    self.poleMass = 5

class InvertedDoublePendulumBulletEnv(MJCFBaseBulletEnv):

  def __init__(self):
    self.robot = InvertedDoublePendulum()
    MJCFBaseBulletEnv.__init__(self, self.robot)
    self.stateId = -1

    self.torqueForce = 200 # In double pendulum the initial torqueForce is 200
    self.gravity = 9.8
    self.pole1Mass = 5
    self.pole2Mass = 5
  
  def get_features(self):
    return self.torqueForce, self.gravity, self.pole1Mass, self.pole2Mass
  
  def set_features(self, features = [200, 9.8, 5, 5]):
    self.reset()
    
    torqueForce = features[0]
    gravity = features[1]
    pole1Mass = features[2] 
    pole2Mass = features[3]

    self.torqueForce = torqueForce
    self.gravity = gravity
    self.pole1Mass = pole1Mass
    self.pole2Mass = pole2Mass

    self._p.setGravity(0, 0, -self.gravity)

    bodyIndex = self.robot.parts["pole"].bodyIndex
    bodyUniqueId = self.robot.parts["pole"].bodies[bodyIndex]
    partIndex = self.robot.parts["pole"].bodyPartIndex
    self._p.changeDynamics(bodyUniqueId, partIndex, mass = self.pole1Mass)

    bodyIndex = self.robot.parts["pole2"].bodyIndex
    bodyUniqueId = self.robot.parts["pole2"].bodies[bodyIndex]
    partIndex = self.robot.parts["pole2"].bodyPartIndex
    self._p.changeDynamics(bodyUniqueId, partIndex, mass = self.pole2Mass)

  def randomize(self, level=0):
    self.reset()

    if(level == 0):
      self.torqueForce = self.np_random.uniform(low=180, high=220)
      self.gravity = self.np_random.uniform(low=8.82, high=10.78)
      self.pole1Mass = self.np_random.uniform(low=4.5, high=5.5)
      self.pole2Mass = self.np_random.uniform(low=4.5, high=5.5)
    elif(level == 1):
      if(self.np_random.uniform(low=0, high=1.0) > 0.5):
        self.torqueForce = self.np_random.uniform(low=50, high=75)
      else:
        self.torqueForce = self.np_random.uniform(low=125, high=150)
      
      if(self.np_random.uniform(low=0, high=1.0) > 0.5):
        self.gravity = self.np_random.uniform(low=4.9, high=7.35)
      else:
        self.gravity = self.np_random.uniform(low=12.25, high=14.7)

      if(self.np_random.uniform(low=0, high=1.0) > 0.5):
        self.pole1Mass = self.np_random.uniform(low=2.5, high=3.75)
      else:
        self.pole1Mass = self.np_random.uniform(low=6.25, high=7.5)
      
      if(self.np_random.uniform(low=0, high=1.0) > 0.5):
        self.pole2Mass = self.np_random.uniform(low=2.5, high=3.75)
      else:
        self.pole2Mass = self.np_random.uniform(low=6.25, high=7.5)

    else:
      print("ERROR: Invalid randomization mode selected, only 0 or 1 are available")
    
    #print("randomizing, torque: ", self.torqueForce  , " gravity ", self.gravity)

    self._p.setGravity(0, 0, -self.gravity)

    bodyIndex = self.robot.parts["pole"].bodyIndex
    bodyUniqueId = self.robot.parts["pole"].bodies[bodyIndex]
    partIndex = self.robot.parts["pole"].bodyPartIndex

    self._p.changeDynamics(bodyUniqueId, partIndex, mass = self.pole1Mass)

    bodyIndex = self.robot.parts["pole2"].bodyIndex
    bodyUniqueId = self.robot.parts["pole2"].bodies[bodyIndex]
    partIndex = self.robot.parts["pole2"].bodyPartIndex

    self._p.changeDynamics(bodyUniqueId, partIndex, mass = self.pole2Mass)

  def create_single_player_scene(self, bullet_client):
    return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

  def reset(self):

    if (self.stateId >= 0):
      self._p.restoreState(self.stateId)
    r = MJCFBaseBulletEnv.reset(self)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()

    self._p.setGravity(0, 0, -self.gravity)

    bodyIndex = self.robot.parts["pole"].bodyIndex
    bodyUniqueId = self.robot.parts["pole"].bodies[bodyIndex]
    partIndex = self.robot.parts["pole"].bodyPartIndex
    self._p.changeDynamics(bodyUniqueId, partIndex, mass = self.pole1Mass)

    bodyIndex = self.robot.parts["pole2"].bodyIndex
    bodyUniqueId = self.robot.parts["pole2"].bodies[bodyIndex]
    partIndex = self.robot.parts["pole2"].bodyPartIndex
    self._p.changeDynamics(bodyUniqueId, partIndex, mass = self.pole2Mass)
    
    return r
  
  def apply_action(self, a):
    assert (np.isfinite(a).all())
    self.robot.slider.set_motor_torque(self.torqueForce * float(np.clip(a[0], -1, +1)))

  def step(self, a):
    #print("GLOBAL: torqueForce: ", self.torqueForce, " gravity: ", self.gravity, " pole1Mass: ", self.pole1Mass, " pole2Mass: ", self.pole2Mass)
    #self.robot.apply_action(a)
    self.apply_action(a)
    self.scene.global_step()
    state = self.robot.calc_state()  # sets self.pos_x self.pos_y
    # upright position: 0.6 (one pole) + 0.6 (second pole) * 0.5 (middle of second pole) = 0.9
    # using <site> tag in original xml, upright position is 0.6 + 0.6 = 1.2, difference +0.3
    dist_penalty = 0.01 * self.robot.pos_x**2 + (self.robot.pos_y + 0.3 - 2)**2
    # v1, v2 = self.model.data.qvel[1:3] TODO when this fixed https://github.com/bulletphysics/bullet3/issues/1040
    #vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
    vel_penalty = 0
    alive_bonus = 10
    done = self.robot.pos_y + 0.3 <= 1
    self.rewards = [float(alive_bonus), float(-dist_penalty), float(-vel_penalty)]
    self.HUD(state, a, done)
    return state, sum(self.rewards), done, {}

  def camera_adjust(self):
    self.camera.move_and_look_at(0, 1.2, 1.2, 0, 0, 0.5)
