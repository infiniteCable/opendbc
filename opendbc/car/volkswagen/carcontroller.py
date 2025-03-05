import math
import numpy as np
from opendbc.can.packer import CANPacker
from opendbc.car import Bus, DT_CTRL, apply_driver_steer_torque_limits, apply_std_steer_angle_limits, structs
from opendbc.car.common.pt2 import PT2Filter
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.volkswagen import mqbcan, pqcan, mebcan
from opendbc.car.volkswagen.values import CANBUS, CarControllerParams, VolkswagenFlags

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState


def get_long_jerk_limits(accel: float, accel_last: float, a_ego: float, dt: float, jerk_prev: float, override: bool):
  # jerk limit are used to improve comfort
  # override mechanics reminder:
  # (1) sending accel = 0 and directly setting jerk to zero results in round about steady accel until harder accel pedal press -> lack of control
  # (2) sending accel = 0 and allowing a high jerk results in a abrupt accel cut -> lack of comfort
  # -> set comfortable jerks
  jerk_limit = 5.0
  jerk_limit_min = 0.5
  factor_up = 2.0
  factor_down = 3.0
  error_gain = 0.6

  if override:
    jerk_raw = 0.
    jerk_up = jerk_limit_min
    jerk_down = jerk_limit_min
  else:
    accel_diff = (accel - accel_last) / dt
    jerk_raw = 0.9 * jerk_prev + 0.1 * accel_diff
    a_error = accel - a_ego
    jerk_raw += a_error * error_gain
    jerk_up = jerk_raw * factor_up
    jerk_down = -jerk_raw * factor_down
    jerk_up = max(jerk_limit_min, min(jerk_up, jerk_limit))
    jerk_down = max(jerk_limit_min, min(jerk_down, jerk_limit))

  return jerk_up, jerk_down, jerk_raw


def get_long_control_limits(speed: float, set_speed: float, distance: float):
  # control limits are used to improve comfort
  # also used to reduce an effect of decel overshoot when target is breaking
  # limits are controlled mainly by distance of lead car
  # problem: no data for approching a non car like target: for now keep limits at minimum if no lead is detected   
  lower_limit_factor = 0.048
  lower_limit_min = 0.
  lower_limit_max = lower_limit_factor * 6
  upper_limit_factor = 0.0625
  upper_limit_min = 0.
  upper_limit_max = upper_limit_factor * 2
  
  upper_limit = np.interp(distance, [0, 100], [upper_limit_min, upper_limit_max]) # base line based on distance
  
  set_speed_diff_up = max(0, abs(speed) - abs(set_speed)) # set speed difference down requested by user or speed overshoot (includes hud - real speed difference!)
  set_speed_diff_up_factor = np.interp(set_speed_diff_up, [1, 1.75], [1., 0.]) # faster requested speed decrease and less speed overshoot downhill 
  lower_limit = np.interp(distance, [0, 6, 100], [lower_limit_min, lower_limit_factor, lower_limit_max]) # base line based on distance
  lower_limit = lower_limit * set_speed_diff_up_factor
  
  return upper_limit, lower_limit


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.CCP = CarControllerParams(CP)
    if CP.flags & VolkswagenFlags.PQ:
      self.CCS = pqcan
    elif CP.flags & VolkswagenFlags.MEB:
      self.CCS = mebcan
    else:
      self.CCS = mqbcan
    self.packer_pt = CANPacker(dbc_names[Bus.pt])
    self.ext_bus = CANBUS.pt if CP.networkLocation == structs.CarParams.NetworkLocation.fwdCamera else CANBUS.cam
    self.aeb_available = not CP.flags & VolkswagenFlags.PQ

    self.apply_torque_last = 0
    self.apply_curvature_last = 0
    self.steering_power_last = 0
    self.accel_last = 0
    self.long_jerk_last = 0
    self.long_override_counter = 0
    self.long_disabled_counter = 0
    self.gra_acc_counter_last = None
    self.eps_timer_soft_disable_alert = False
    self.hca_frame_timer_running = 0
    self.hca_frame_same_torque = 0
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0
    self.smooth_curv = PT2Filter(46.0, 1.0, self.CCP.STEER_STEP * DT_CTRL) # effectivly adds a small delay, compensate with steering actuator delay)

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    can_sends = []

    # **** Steering Controls ************************************************ #

    if self.frame % self.CCP.STEER_STEP == 0:
      if self.CP.flags & VolkswagenFlags.MEB:
        # Logic to avoid HCA refused state:
        #   * steering power as counter and near zero before OP lane assist deactivation
        # MEB rack can be used continously without time limits
        # maximum real steering angle change ~ 120-130 deg/s

        if CC.latActive:
          hca_enabled = True
          current_curvature = CS.curvature
          actuator_curvature_with_offset = actuators.curvature + (CS.curvature - CC.currentCurvature)
          apply_curvature = self.smooth_curv.update(actuator_curvature_with_offset) # reduce wear, better comfort and car stability without reducing steering ability
          apply_curvature = apply_std_steer_angle_limits(apply_curvature, self.apply_curvature_last, CS.out.vEgoRaw, 0., CC.latActive, self.CCP.ANGLE_LIMITS)
          if CS.out.steeringPressed: # roughly sync curvature when user overrides
            apply_curvature = np.clip(apply_curvature, current_curvature - self.CCP.CURVATURE_ERROR, current_curvature + self.CCP.CURVATURE_ERROR)
          apply_curvature = np.clip(apply_curvature, -self.CCP.CURVATURE_MAX, self.CCP.CURVATURE_MAX)

          steering_power_min_by_speed = np.interp(CS.out.vEgoRaw, [0, self.CCP.STEERING_POWER_MAX_BY_SPEED], [self.CCP.STEERING_POWER_MIN, self.CCP.STEERING_POWER_MAX]) # base level
          steering_curvature_diff = abs(apply_curvature - current_curvature) # keep power high at very low speed for both directions
          steering_curvature_increase = max(0, abs(apply_curvature) - abs(current_curvature)) # increase power for increasing steering at normal driving speeds
          steering_curvature_change = np.interp(CS.out.vEgoRaw, [0., 3.], [steering_curvature_diff, steering_curvature_increase]) # maximum power seems to inhibit steering movement, decreasing does not increase power
          steering_power_target_curvature = steering_power_min_by_speed + self.CCP.CURVATURE_POWER_FACTOR * (steering_curvature_change + abs(apply_curvature)) # abs apply_curvature level keeps steering in place
          steering_power_target = np.clip(steering_power_target_curvature, self.CCP.STEERING_POWER_MIN, self.CCP.STEERING_POWER_MAX)

          if self.steering_power_last < self.CCP.STEERING_POWER_MIN:  # OP lane assist just activated
            steering_power = min(self.steering_power_last + self.CCP.STEERING_POWER_STEPS, self.CCP.STEERING_POWER_MIN)
          elif CS.out.steeringPressed:  # user action results in decreasing the steering power
            steering_power_user = max(steering_power_target / 100 * (100 - self.CCP.STEERING_POWER_USER_REDUCTION), self.CCP.STEERING_POWER_MIN)
            steering_power = max(self.steering_power_last - self.CCP.STEERING_POWER_STEPS, steering_power_user)
          else: # following desired target
            if self.steering_power_last < steering_power_target:
              steering_power = min(self.steering_power_last + self.CCP.STEERING_POWER_STEPS, steering_power_target)
            elif self.steering_power_last > steering_power_target:
              steering_power = max(self.steering_power_last - self.CCP.STEERING_POWER_STEPS, steering_power_target)
            else:
              steering_power = self.steering_power_last

          steering_power_boost = True if steering_power == self.CCP.STEERING_POWER_MAX else False
          
        else:
          steering_power_boost = False
          if self.steering_power_last > 0: # keep HCA alive until steering power has reduced to zero
            hca_enabled = True
            current_curvature = CS.curvature
            apply_curvature = np.clip(current_curvature, -self.CCP.CURVATURE_MAX, self.CCP.CURVATURE_MAX) # synchronize with current curvature
            steering_power = max(self.steering_power_last - self.CCP.STEERING_POWER_STEPS, 0)
          else: 
            hca_enabled = False
            apply_curvature = 0. # inactive curvature
            steering_power = 0

        can_sends.append(self.CCS.create_steering_control(self.packer_pt, CANBUS.pt, apply_curvature, hca_enabled, steering_power, steering_power_boost))
        self.apply_curvature_last = apply_curvature
        self.steering_power_last = steering_power
        
      else:
        # Logic to avoid HCA state 4 "refused":
        #   * Don't steer unless HCA is in state 3 "ready" or 5 "active"
        #   * Don't steer at standstill
        #   * Don't send > 3.00 Newton-meters torque
        #   * Don't send the same torque for > 6 seconds
        #   * Don't send uninterrupted steering for > 360 seconds
        # MQB racks reset the uninterrupted steering timer after a single frame
        # of HCA disabled; this is done whenever output happens to be zero.

        if CC.latActive:
          new_torque = int(round(actuators.torque * self.CCP.STEER_MAX))
          apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorque, self.CCP)
          self.hca_frame_timer_running += self.CCP.STEER_STEP
          if self.apply_torque_last == apply_torque:
            self.hca_frame_same_torque += self.CCP.STEER_STEP
            if self.hca_frame_same_torque > self.CCP.STEER_TIME_STUCK_TORQUE / DT_CTRL:
              apply_torque -= (1, -1)[apply_torque < 0]
              self.hca_frame_same_torque = 0
          else:
            self.hca_frame_same_torque = 0
          hca_enabled = abs(apply_torque) > 0
        else:
          hca_enabled = False
          apply_torque = 0

        if not hca_enabled:
          self.hca_frame_timer_running = 0

        self.eps_timer_soft_disable_alert = self.hca_frame_timer_running > self.CCP.STEER_TIME_ALERT / DT_CTRL
        self.apply_torque_last = apply_torque
        can_sends.append(self.CCS.create_steering_control(self.packer_pt, CANBUS.pt, apply_torque, hca_enabled))

      if self.CP.flags & VolkswagenFlags.STOCK_HCA_PRESENT:
        # Pacify VW Emergency Assist driver inactivity detection by changing its view of driver steering input torque
        # to the greatest of actual driver input or 2x openpilot's output (1x openpilot output is not enough to
        # consistently reset inactivity detection on straight level roads). See commaai/openpilot#23274 for background.
        ea_simulated_torque = float(np.clip(apply_torque * 2, -self.CCP.STEER_MAX, self.CCP.STEER_MAX))
        if abs(CS.out.steeringTorque) > abs(ea_simulated_torque):
          ea_simulated_torque = CS.out.steeringTorque
        can_sends.append(self.CCS.create_eps_update(self.packer_pt, CANBUS.cam, CS.eps_stock_values, ea_simulated_torque))

    # by jyoung anti EA intervention, send default values
    if self.CP.flags & VolkswagenFlags.MEB:
      if self.frame % 2 == 0:
        can_sends.append(mebcan.create_ea_control(self.packer_pt, CANBUS.pt))
      if self.frame % 50 == 0:
        can_sends.append(mebcan.create_ea_hud(self.packer_pt, CANBUS.pt))

    # **** Acceleration Controls ******************************************** #

    if self.frame % self.CCP.ACC_CONTROL_STEP == 0 and self.CP.openpilotLongitudinalControl:
      stopping = actuators.longControlState == LongCtrlState.stopping
      starting = actuators.longControlState == LongCtrlState.pid and (CS.esp_hold_confirmation or CS.out.vEgo < self.CP.vEgoStopping)

      if self.CP.flags & VolkswagenFlags.MEB:
        # Logic to prevent car error with EPB:
        #   * send a few frames of HMS RAMP RELEASE command at the very begin of long override
        #   * send a few frames of HMS RAMP RELEASE command right at the end of active long control
        accel = float(np.clip(actuators.accel, self.CCP.ACCEL_MIN, self.CCP.ACCEL_MAX) if CC.enabled else 0)

        # 1 frame of long_override_begin is enough, but lower the possibility of panda safety blocking it for now until we adapt panda safety correctly
        long_override = CC.cruiseControl.override or CS.out.gasPressed
        self.long_override_counter = min(self.long_override_counter + 1, 5) if long_override else 0
        long_override_begin = long_override and self.long_override_counter < 5

        # 1 frame of long_disabling is enough, but lower the possibility of panda safety blocking it for now until we adapt panda safety correctly
        self.long_disabled_counter = min(self.long_disabled_counter + 1, 5) if not CC.enabled else 0
        long_disabling = not CC.enabled and self.long_disabled_counter < 5

        upper_control_limit, lower_control_limit = get_long_control_limits(CS.out.vEgoRaw, hud_control.setSpeed, hud_control.leadDistance) if CC.enabled else (0, 0)
        upper_jerk, lower_jerk, self.long_jerk_last = get_long_jerk_limits(accel, self.accel_last, CS.out.aEgo, DT_CTRL * self.CCP.ACC_CONTROL_STEP, self.long_jerk_last, long_override) if CC.enabled else (0, 0, 0)
        
        acc_control = self.CCS.acc_control_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.enabled,
                                                 CS.esp_hold_confirmation, long_override)          
        acc_hold_type = self.CCS.acc_hold_type(CS.out.cruiseState.available, CS.out.accFaulted, CC.enabled, starting, stopping,
                                               CS.esp_hold_confirmation, long_override, long_override_begin, long_disabling)
        can_sends.extend(self.CCS.create_acc_accel_control(self.packer_pt, CANBUS.pt, CS.acc_type, CC.enabled,
                                                           upper_jerk, lower_jerk, upper_control_limit, lower_control_limit,
                                                           accel, acc_control, acc_hold_type, stopping, starting, CS.esp_hold_confirmation,
                                                           long_override, CS.travel_assist_available))
        self.accel_last = accel

      else:
        accel = float(np.clip(actuators.accel, self.CCP.ACCEL_MIN, self.CCP.ACCEL_MAX) if CC.longActive else 0)
        self.accel_last = accel
        
        acc_control = self.CCS.acc_control_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.longActive)
        can_sends.extend(self.CCS.create_acc_accel_control(self.packer_pt, CANBUS.pt, CS.acc_type, CC.longActive, accel,
                                                           acc_control, stopping, starting, CS.esp_hold_confirmation))

      #if self.aeb_available:
      #  if self.frame % self.CCP.AEB_CONTROL_STEP == 0:
      #    can_sends.append(self.CCS.create_aeb_control(self.packer_pt, False, False, 0.0))
      #  if self.frame % self.CCP.AEB_HUD_STEP == 0:
      #    can_sends.append(self.CCS.create_aeb_hud(self.packer_pt, False, False))

    # **** HUD Controls ***************************************************** #

    if self.frame % self.CCP.LDW_STEP == 0:
      hud_alert = 0
      
      if hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw):
        hud_alert = self.CCP.LDW_MESSAGES["laneAssistTakeOverUrgent"]
        
      if self.CP.flags & VolkswagenFlags.MEB:
        sound_alert = self.CCP.LDW_SOUNDS["Beep"] if hud_alert == self.CCP.LDW_MESSAGES["laneAssistTakeOverUrgent"] else self.CCP.LDW_SOUNDS["None"]
        can_sends.append(self.CCS.create_lka_hud_control(self.packer_pt, CANBUS.pt, CS.ldw_stock_values, CC.latActive,
                                                         CS.out.steeringPressed, hud_alert, hud_control, sound_alert))
      else:
        can_sends.append(self.CCS.create_lka_hud_control(self.packer_pt, CANBUS.pt, CS.ldw_stock_values, CC.latActive,
                                                         CS.out.steeringPressed, hud_alert, hud_control))

    if hud_control.leadDistanceBars != self.lead_distance_bars_last:
      self.distance_bar_frame = self.frame

    if self.frame % self.CCP.ACC_HUD_STEP == 0 and self.CP.openpilotLongitudinalControl:
      if self.CP.flags & VolkswagenFlags.MEB:
        fcw_alert = True if hud_control.visualAlert == VisualAlert.fcw else False
        show_distance_bars = self.frame - self.distance_bar_frame < 400
        gap = max(8, CS.out.vEgo * hud_control.leadFollowTime)
        distance = max(8, hud_control.leadDistance) if hud_control.leadDistance != 0 else 0
        acc_hud_status = self.CCS.acc_hud_status_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.enabled,
                                                       CS.esp_hold_confirmation, CC.cruiseControl.override or CS.out.gasPressed)
        can_sends.append(self.CCS.create_acc_hud_control(self.packer_pt, CANBUS.pt, acc_hud_status, hud_control.setSpeed * CV.MS_TO_KPH,
                                                         hud_control.leadVisible, hud_control.leadDistanceBars + 1, show_distance_bars,
                                                         CS.esp_hold_confirmation, distance, gap, fcw_alert))

      else:
        lead_distance = 0
        if hud_control.leadVisible and self.frame * DT_CTRL > 1.0:  # Don't display lead until we know the scaling factor
          lead_distance = 512 if CS.upscale_lead_car_signal else 8
        acc_hud_status = self.CCS.acc_hud_status_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.longActive)
        # FIXME: follow the recent displayed-speed updates, also use mph_kmh toggle to fix display rounding problem?
        set_speed = hud_control.setSpeed * CV.MS_TO_KPH
        can_sends.append(self.CCS.create_acc_hud_control(self.packer_pt, CANBUS.pt, acc_hud_status, set_speed,
                                                         lead_distance, hud_control.leadDistanceBars))

    # **** Stock ACC Button Controls **************************************** #

    gra_send_ready = self.CP.pcmCruise and CS.gra_stock_values["COUNTER"] != self.gra_acc_counter_last
    if gra_send_ready and (CC.cruiseControl.cancel or CC.cruiseControl.resume):
      can_sends.append(self.CCS.create_acc_buttons_control(self.packer_pt, self.ext_bus, CS.gra_stock_values,
                                                           cancel=CC.cruiseControl.cancel, resume=CC.cruiseControl.resume))

    new_actuators = actuators.as_builder()
    new_actuators.torque = self.apply_torque_last / self.CCP.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last
    new_actuators.curvature = float(self.apply_curvature_last)
    new_actuators.accel = self.accel_last

    self.lead_distance_bars_last = hud_control.leadDistanceBars
    self.gra_acc_counter_last = CS.gra_stock_values["COUNTER"]
    self.frame += 1
    return new_actuators, can_sends
