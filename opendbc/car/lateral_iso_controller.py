import numpy as np
from opendbc.car import DT_CTRL

# Konstanten für ISO 11270 Limitierung
MIN_SPEED = 1.0
MAX_CURVATURE = 0.2
MAX_LATERAL_JERK = 5.0
ISO_LATERAL_ACCEL = 3.0
EARTH_G = 9.81
AVERAGE_ROAD_ROLL = 0.06  # Statische Roll-Mechanik (~3.4° Neigung, 6% Superelevation)
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (EARTH_G * AVERAGE_ROAD_ROLL)  # ~2.4 m/s^2

SOFT_LIMIT_TIME = 2.0


class LateralISOController:
  def __init__(self, steer_step):
    self.steer_step_time = steer_step * DT_CTRL
    self.prev_curvature = 0.0
    self.soft_limit_active = False
    self.soft_limit_timer = 0.0
    self.soft_limit_counter = 0
    self.soft_limit_start_curvature = 0.0
    self.soft_limit_steps = int(SOFT_LIMIT_TIME / self.steer_step_time)
  
  def update(self, v_ego, new_curvature, steering_pressed):
    v_ego = max(v_ego, MIN_SPEED)
    max_curvature_rate = MAX_LATERAL_JERK / (v_ego ** 2)
    
    new_curvature = np.clip(new_curvature,
                            self.prev_curvature - max_curvature_rate * self.steer_step_time,
                            self.prev_curvature + max_curvature_rate * self.steer_step_time)
    
    max_lat_accel = MAX_LATERAL_ACCEL
    min_lat_accel = -MAX_LATERAL_ACCEL

    iso_limit_exceeded = abs(new_curvature * v_ego ** 2) > max_lat_accel

    if not iso_limit_exceeded:
      self.soft_limit_active = False
    
    if steering_pressed and iso_limit_exceeded:
      self.soft_limit_active = True
      self.soft_limit_counter = 0
      self.soft_limit_start_curvature = self.prev_curvature

    if self.soft_limit_active:
      self.soft_limit_counter += 1
      alpha = min(1, self.soft_limit_counter / self.soft_limit_steps)
      target_curvature = (1 - alpha) * self.soft_limit_start_curvature + alpha * new_curvature
      new_curvature = target_curvature

      if self.soft_limit_counter >= self.soft_limit_steps:
        self.soft_limit_active = False
    
    else:
      new_curvature = np.clip(new_curvature, min_lat_accel / v_ego ** 2, max_lat_accel / v_ego ** 2)
    
    new_curvature, limited_max_curv = np.clip(new_curvature, -MAX_CURVATURE, MAX_CURVATURE), abs(new_curvature) > MAX_CURVATURE
    
    self.prev_curvature = new_curvature
    return float(new_curvature), limited_max_curv or iso_limit_exceeded
