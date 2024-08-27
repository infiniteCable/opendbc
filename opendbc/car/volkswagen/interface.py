from panda import Panda
from opendbc.car import get_safety_config, structs
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.volkswagen.values import CAR, NetworkLocation, TransmissionType, VolkswagenFlags


class CarInterface(CarInterfaceBase):
  @staticmethod
  def _get_params(ret: structs.CarParams, candidate: CAR, fingerprint, car_fw, experimental_long, docs) -> structs.CarParams:
    ret.carName = "volkswagen"
    ret.radarUnavailable = True

    if ret.flags & VolkswagenFlags.PQ:
      # Set global PQ35/PQ46/NMS parameters
      ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.volkswagenPq)]
      ret.enableBsm = 0x3BA in fingerprint[0]  # SWA_1

      if 0x440 in fingerprint[0] or docs:  # Getriebe_1
        ret.transmissionType = TransmissionType.automatic
      else:
        ret.transmissionType = TransmissionType.manual

      if any(msg in fingerprint[1] for msg in (0x1A0, 0xC2)):  # Bremse_1, Lenkwinkel_1
        ret.networkLocation = NetworkLocation.gateway
      else:
        ret.networkLocation = NetworkLocation.fwdCamera

      # The PQ port is in dashcam-only mode due to a fixed six-minute maximum timer on HCA steering. An unsupported
      # EPS flash update to work around this timer, and enable steering down to zero, is available from:
      #   https://github.com/pd0wm/pq-flasher
      # It is documented in a four-part blog series:
      #   https://blog.willemmelching.nl/carhacking/2022/01/02/vw-part1/
      # Panda ALLOW_DEBUG firmware required.
      ret.dashcamOnly = True
      
    elif ret.flags & VolkswagenFlags.MEB: # TODO
      # Set global MEB parameters
      ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.volkswagenMeb)]
      ret.enableBsm        = True
      ret.transmissionType = TransmissionType.direct
      ret.networkLocation  = NetworkLocation.fwdCamera # TODO signal sources: I am connected at gateway/ICAS 1 right now
      ret.steerControlType = structs.CarParams.SteerControlType.angle
      ret.radarUnavailable = False
      #ret.flags |= VolkswagenFlags.STOCK_HCA_PRESENT.value

      if any(msg in fingerprint[1] for msg in (0x520, 0x86, 0xFD, 0x13D)):  # Airbag_02, LWI_01, ESP_21, MEB_EPS_01
        ret.networkLocation = NetworkLocation.gateway
      else:
        ret.networkLocation = NetworkLocation.fwdCamera

    else:
      # Set global MQB parameters
      ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.volkswagen)]
      ret.enableBsm = 0x30F in fingerprint[0]  # SWA_01

      if 0xAD in fingerprint[0] or docs:  # Getriebe_11
        ret.transmissionType = TransmissionType.automatic
      elif 0x187 in fingerprint[0]:  # EV_Gearshift
        ret.transmissionType = TransmissionType.direct
      else:
        ret.transmissionType = TransmissionType.manual

      if any(msg in fingerprint[1] for msg in (0x40, 0x86, 0xB2, 0xFD)):  # Airbag_01, LWI_01, ESP_19, ESP_21
        ret.networkLocation = NetworkLocation.gateway
      else:
        ret.networkLocation = NetworkLocation.fwdCamera

      if 0x126 in fingerprint[2]:  # HCA_01
        ret.flags |= VolkswagenFlags.STOCK_HCA_PRESENT.value

    # Global lateral tuning defaults, can be overridden per-vehicle

    if ret.flags & VolkswagenFlags.PQ:
      ret.steerLimitTimer = 0.4
      ret.steerActuatorDelay = 0.2
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)
    elif ret.flags & VolkswagenFlags.MEB:
      ret.steerLimitTimer = 0.8
      ret.steerActuatorDelay = 0.3
    else:
      ret.steerLimitTimer = 0.4
      ret.steerActuatorDelay = 0.1
      ret.lateralTuning.pid.kpBP = [0.]
      ret.lateralTuning.pid.kiBP = [0.]
      ret.lateralTuning.pid.kf = 0.00006
      ret.lateralTuning.pid.kpV = [0.6]
      ret.lateralTuning.pid.kiV = [0.2]

    # Global longitudinal tuning defaults, can be overridden per-vehicle

    if ret.flags & VolkswagenFlags.MEB:
      ret.longitudinalActuatorDelay = 0.5
      #ret.longitudinalTuning.deadzoneBP = [0., 8.05]
      #ret.longitudinalTuning.deadzoneV = [.0, .14]
      ret.longitudinalTuning.kpBP = [0., 5., 20.]
      ret.longitudinalTuning.kpV  = [0.1, 0.05, 0.]
      ret.longitudinalTuning.kiBP = [0., 5., 20.]
      ret.longitudinalTuning.kiV  = [0., 0., -0.13]
      #if params.get_bool('ExperimentalMode'):
      #  ret.longitudinalTuning.kpV = [0.5, 0.2, -0.2] # experimental OP long is less smooth
      
    ret.experimentalLongitudinalAvailable = ret.networkLocation == NetworkLocation.gateway or docs
    if experimental_long:
      # Proof-of-concept, prep for E2E only. No radar points available for non MEB. Panda ALLOW_DEBUG firmware required.
      ret.openpilotLongitudinalControl = True
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_VOLKSWAGEN_LONG_CONTROL
      if ret.transmissionType == TransmissionType.manual:
        ret.minEnableSpeed = 4.5

    ret.vEgoStarting = 0.1
    ret.vEgoStopping = 0.5
    ret.pcmCruise = not ret.openpilotLongitudinalControl
    ret.stoppingControl = True
    ret.stopAccel = -0.55
    ret.autoResumeSng = ret.minEnableSpeed == -1

    return ret