import numpy as np


def get_long_jerk_limits(enabled, override, accel, accel_last, jerk_up, jerk_down, dy_up, dy_down, dt,
                         critical_state, filter_gain=0.9, jerk_limit_min=0.5, jerk_limit_max=5.0):
  # jerk limits by accel change are used to improve comfort while ensuring a fast enough car reaction
  # override mechanics reminder:
  # (1) sending accel = 0 and directly setting jerk to zero results in round about steady accel until harder accel pedal press -> lack of control
  # (2) sending accel = 0 and allowing a high jerk results in a abrupt accel cut -> lack of comfort
  if not enabled:
    return 0., 0., 0., 0.

  if override:
    jerk_up = jerk_limit_min
    jerk_down = jerk_limit_min
    dy_up = 0.
    dy_down = 0.
  elif critical_state: # force best car reaction
    jerk_up = jerk_limit_max
    jerk_down = jerk_limit_max
    dy_up = 0.
    dy_down = 0.
  else:
    j = (accel - accel_last) / dt

    tgt_up = abs(j) if j > 0 else 0.
    tgt_down = abs(j) if j < 0 else 0.

    # how fast does the car react to acceleration
    dy_up += filter_gain * (tgt_up - jerk_up - dy_up)
    jerk_up += dt * dy_up
    jerk_up = np.clip(jerk_up, jerk_limit_min, jerk_limit_max)

    # how fast does the car react to braking
    dy_down += filter_gain * (tgt_down - jerk_down - dy_down)
    jerk_down += dt * dy_down
    jerk_down = np.clip(jerk_down, jerk_limit_min, jerk_limit_max)

  return jerk_up, jerk_down, dy_up, dy_down


def get_long_control_limits(enabled: bool, speed: float, set_speed: float, distance: float, critical_state: bool):
  # control limits by distance are used to improve comfort while ensuring precise car reaction if neccessary
  # also used to reduce an effect of decel overshoot when target is breaking
  # limits are controlled mainly by distance of lead car
  # problem: no data for approching a non car like target: for now keep limits at minimum if no lead is detected   
  if not enabled:
    return 0., 0.

  lower_limit_factor = 0.048
  lower_limit_min = 0.
  lower_limit_max = lower_limit_factor * 6
  upper_limit_factor = 0.0625
  upper_limit_min = 0.
  upper_limit_max = upper_limit_factor * 2

  if critical_state: # force most precise accel command execution
    return lower_limit_min, upper_limit_min

  # how far can the true accel vary downwards from requested accel
  upper_limit = np.interp(distance, [0, 100], [upper_limit_min, upper_limit_max]) # base line based on distance

  # how far can the true accel vary upwards from requested accel
  set_speed_diff_up = max(0, abs(speed) - abs(set_speed)) # set speed difference down requested by user or speed overshoot (includes hud - real speed difference!)
  set_speed_diff_up_factor = np.interp(set_speed_diff_up, [1, 1.75], [1., 0.]) # faster requested speed decrease and less speed overshoot downhill 
  lower_limit = np.interp(distance, [0, 100], [lower_limit_min, lower_limit_max]) # base line based on distance
  lower_limit = lower_limit * set_speed_diff_up_factor

  return upper_limit, lower_limit


def sigmoid_curvature_boost_meb(kappa: float, v_ego: float, kappa_thresh: float = 0.0) -> float:
  # compensate non linear behaviour: boost low curvatures
  # this is either a model issue (nerfing low curvatures) or a specific steering rack behaviour
  v_points = np.array([20.0, 40.0])
  boost_values = np.array([1.5, 2.1]) # increase boost amplitude with speed
  boost = float(np.interp(v_ego, v_points, boost_values))
  steepness_values = np.array([5000.0, 3200.0]) # increase boost area with speed
  steepness = float(np.interp(v_ego, v_points, steepness_values))

  abs_kappa = abs(kappa)
  boost_factor = 1.0 + (boost - 1.0) / (1 + np.exp(steepness * (abs_kappa - kappa_thresh)))

  return np.sign(kappa) * abs_kappa * boost_factor
