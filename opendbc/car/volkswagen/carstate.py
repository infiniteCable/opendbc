import numpy as np
from opendbc.can.parser import CANParser
from opendbc.car import Bus, structs
from opendbc.car.interfaces import CarStateBase
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.volkswagen.values import DBC, CANBUS, NetworkLocation, TransmissionType, GearShifter, \
                                                      CarControllerParams, VolkswagenFlags

ButtonType = structs.CarState.ButtonEvent.Type


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.frame = 0
    self.eps_init_complete = False
    self.CCP = CarControllerParams(CP)
    self.button_states = {button.event_type: False for button in self.CCP.BUTTONS}
    self.esp_hold_confirmation = False
    self.upscale_lead_car_signal = False
    self.eps_stock_values = False
    self.curvature = 0.

  def update_button_enable(self, buttonEvents: list[structs.CarState.ButtonEvent]):
    if not self.CP.pcmCruise:
      for b in buttonEvents:
        # Enable OP long on falling edge of enable buttons
        if b.type in (ButtonType.setCruise, ButtonType.resumeCruise) and not b.pressed:
          return True
    return False

  def create_button_events(self, pt_cp, buttons):
    button_events = []

    for button in buttons:
      state = pt_cp.vl[button.can_addr][button.can_msg] in button.values
      if self.button_states[button.event_type] != state:
        event = structs.CarState.ButtonEvent()
        event.type = button.event_type
        event.pressed = state
        button_events.append(event)
      self.button_states[button.event_type] = state

    return button_events

  def update(self, can_parsers) -> structs.CarState:
    pt_cp = can_parsers[Bus.pt]
    cam_cp = can_parsers[Bus.cam]
    ext_cp = pt_cp if self.CP.networkLocation == NetworkLocation.fwdCamera else cam_cp

    if self.CP.flags & VolkswagenFlags.PQ:
      return self.update_pq(pt_cp, cam_cp, ext_cp)
    elif self.CP.flags & VolkswagenFlags.MEB:
      return self.update_meb(pt_cp, cam_cp, ext_cp)

    ret = structs.CarState()

    if self.CP.transmissionType == TransmissionType.direct:
      ret.gearShifter = self.parse_gear_shifter(self.CCP.shifter_values.get(pt_cp.vl["Motor_EV_01"]["MO_Waehlpos"], None))
    elif self.CP.transmissionType == TransmissionType.manual:
      ret.clutchPressed = not pt_cp.vl["Motor_14"]["MO_Kuppl_schalter"]
      if bool(pt_cp.vl["Gateway_72"]["BCM1_Rueckfahrlicht_Schalter"]):
        ret.gearShifter = GearShifter.reverse
      else:
        ret.gearShifter = GearShifter.drive
    else:
      ret.gearShifter = self.parse_gear_shifter(self.CCP.shifter_values.get(pt_cp.vl["Gateway_73"]["GE_Fahrstufe"], None))

    if True:
      # MQB-specific
      self.upscale_lead_car_signal = bool(pt_cp.vl["Kombi_03"]["KBI_Variante"])  # Analog vs digital instrument cluster

      ret.wheelSpeeds = self.get_wheel_speeds(
        pt_cp.vl["ESP_19"]["ESP_VL_Radgeschw_02"],
        pt_cp.vl["ESP_19"]["ESP_VR_Radgeschw_02"],
        pt_cp.vl["ESP_19"]["ESP_HL_Radgeschw_02"],
        pt_cp.vl["ESP_19"]["ESP_HR_Radgeschw_02"],
      )

      ret.yawRate = pt_cp.vl["ESP_02"]["ESP_Gierrate"] * (1, -1)[int(pt_cp.vl["ESP_02"]["ESP_VZ_Gierrate"])] * CV.DEG_TO_RAD
      hca_status = self.CCP.hca_status_values.get(pt_cp.vl["LH_EPS_03"]["EPS_HCA_Status"])
      if self.CP.flags & VolkswagenFlags.STOCK_HCA_PRESENT:
        ret.carFaultedNonCritical = bool(cam_cp.vl["HCA_01"]["EA_Ruckfreigabe"]) or cam_cp.vl["HCA_01"]["EA_ACC_Sollstatus"] > 0  # EA

      drive_mode = True
      ret.gas = pt_cp.vl["Motor_20"]["MO_Fahrpedalrohwert_01"] / 100.0
      ret.brake = pt_cp.vl["ESP_05"]["ESP_Bremsdruck"] / 250.0  # FIXME: this is pressure in Bar, not sure what OP expects
      brake_pedal_pressed = bool(pt_cp.vl["Motor_14"]["MO_Fahrer_bremst"])
      brake_pressure_detected = bool(pt_cp.vl["ESP_05"]["ESP_Fahrer_bremst"])
      ret.brakePressed = brake_pedal_pressed or brake_pressure_detected
      ret.parkingBrake = bool(pt_cp.vl["Kombi_01"]["KBI_Handbremse"])  # FIXME: need to include an EPB check as well

      ret.doorOpen = any([pt_cp.vl["Gateway_72"]["ZV_FT_offen"],
                          pt_cp.vl["Gateway_72"]["ZV_BT_offen"],
                          pt_cp.vl["Gateway_72"]["ZV_HFS_offen"],
                          pt_cp.vl["Gateway_72"]["ZV_HBFS_offen"],
                          pt_cp.vl["Gateway_72"]["ZV_HD_offen"]])

      if self.CP.enableBsm:
        # Infostufe: BSM LED on, Warnung: BSM LED flashing
        ret.leftBlindspot = bool(ext_cp.vl["SWA_01"]["SWA_Infostufe_SWA_li"]) or bool(ext_cp.vl["SWA_01"]["SWA_Warnung_SWA_li"])
        ret.rightBlindspot = bool(ext_cp.vl["SWA_01"]["SWA_Infostufe_SWA_re"]) or bool(ext_cp.vl["SWA_01"]["SWA_Warnung_SWA_re"])

      ret.stockFcw = bool(ext_cp.vl["ACC_10"]["AWV2_Freigabe"])
      ret.stockAeb = bool(ext_cp.vl["ACC_10"]["ANB_Teilbremsung_Freigabe"]) or bool(ext_cp.vl["ACC_10"]["ANB_Zielbremsung_Freigabe"])

      self.acc_type = ext_cp.vl["ACC_06"]["ACC_Typ"]
      self.esp_hold_confirmation = bool(pt_cp.vl["ESP_21"]["ESP_Haltebestaetigung"])
      acc_limiter_mode = ext_cp.vl["ACC_02"]["ACC_Gesetzte_Zeitluecke"] == 0
      speed_limiter_mode = bool(pt_cp.vl["TSK_06"]["TSK_Limiter_ausgewaehlt"])

      ret.cruiseState.available = pt_cp.vl["TSK_06"]["TSK_Status"] in (2, 3, 4, 5)
      ret.cruiseState.enabled = pt_cp.vl["TSK_06"]["TSK_Status"] in (3, 4, 5)
      ret.cruiseState.speed = ext_cp.vl["ACC_02"]["ACC_Wunschgeschw_02"] * CV.KPH_TO_MS if self.CP.pcmCruise else 0
      ret.accFaulted = pt_cp.vl["TSK_06"]["TSK_Status"] in (6, 7)

      ret.leftBlinker = bool(pt_cp.vl["Blinkmodi_02"]["Comfort_Signal_Left"])
      ret.rightBlinker = bool(pt_cp.vl["Blinkmodi_02"]["Comfort_Signal_Right"])

    # Shared logic

    ret.vEgoRaw = float(np.mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr]))
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)

    ret.steeringAngleDeg = pt_cp.vl["LWI_01"]["LWI_Lenkradwinkel"] * (1, -1)[int(pt_cp.vl["LWI_01"]["LWI_VZ_Lenkradwinkel"])]
    ret.steeringRateDeg = pt_cp.vl["LWI_01"]["LWI_Lenkradw_Geschw"] * (1, -1)[int(pt_cp.vl["LWI_01"]["LWI_VZ_Lenkradw_Geschw"])]
    ret.steeringTorque = pt_cp.vl["LH_EPS_03"]["EPS_Lenkmoment"] * (1, -1)[int(pt_cp.vl["LH_EPS_03"]["EPS_VZ_Lenkmoment"])]
    ret.steeringPressed = abs(ret.steeringTorque) > self.CCP.STEER_DRIVER_ALLOWANCE
    ret.steerFaultTemporary, ret.steerFaultPermanent = self.update_hca_state(hca_status, drive_mode)

    ret.gasPressed = ret.gas > 0
    ret.espActive = bool(pt_cp.vl["ESP_21"]["ESP_Eingriff"])
    ret.espDisabled = pt_cp.vl["ESP_21"]["ESP_Tastung_passiv"] != 0
    ret.seatbeltUnlatched = pt_cp.vl["Airbag_02"]["AB_Gurtschloss_FA"] != 3

    ret.standstill = ret.vEgoRaw == 0
    ret.cruiseState.standstill = self.CP.pcmCruise and self.esp_hold_confirmation
    ret.cruiseState.nonAdaptive = acc_limiter_mode or speed_limiter_mode
    if ret.cruiseState.speed > 90:
      ret.cruiseState.speed = 0

    self.eps_stock_values = pt_cp.vl["LH_EPS_03"]
    self.ldw_stock_values = cam_cp.vl["LDW_02"] if self.CP.networkLocation == NetworkLocation.fwdCamera else {}
    self.gra_stock_values = pt_cp.vl["GRA_ACC_01"]

    ret.buttonEvents = self.create_button_events(pt_cp, self.CCP.BUTTONS)

    self.frame += 1
    return ret

  def update_pq(self, pt_cp, cam_cp, ext_cp) -> structs.CarState:
    ret = structs.CarState()
    # Update vehicle speed and acceleration from ABS wheel speeds.
    ret.wheelSpeeds = self.get_wheel_speeds(
      pt_cp.vl["Bremse_3"]["Radgeschw__VL_4_1"],
      pt_cp.vl["Bremse_3"]["Radgeschw__VR_4_1"],
      pt_cp.vl["Bremse_3"]["Radgeschw__HL_4_1"],
      pt_cp.vl["Bremse_3"]["Radgeschw__HR_4_1"],
    )

    # vEgo obtained from Bremse_1 vehicle speed rather than Bremse_3 wheel speeds because Bremse_3 isn't present on NSF
    ret.vEgoRaw = pt_cp.vl["Bremse_1"]["Geschwindigkeit_neu__Bremse_1_"] * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw == 0

    # Update EPS position and state info. For signed values, VW sends the sign in a separate signal.
    ret.steeringAngleDeg = pt_cp.vl["Lenkhilfe_3"]["LH3_BLW"] * (1, -1)[int(pt_cp.vl["Lenkhilfe_3"]["LH3_BLWSign"])]
    ret.steeringRateDeg = pt_cp.vl["Lenkwinkel_1"]["Lenkradwinkel_Geschwindigkeit"] * (1, -1)[int(pt_cp.vl["Lenkwinkel_1"]["Lenkradwinkel_Geschwindigkeit_S"])]
    ret.steeringTorque = pt_cp.vl["Lenkhilfe_3"]["LH3_LM"] * (1, -1)[int(pt_cp.vl["Lenkhilfe_3"]["LH3_LMSign"])]
    ret.steeringPressed = abs(ret.steeringTorque) > self.CCP.STEER_DRIVER_ALLOWANCE
    ret.yawRate = pt_cp.vl["Bremse_5"]["Giergeschwindigkeit"] * (1, -1)[int(pt_cp.vl["Bremse_5"]["Vorzeichen_der_Giergeschwindigk"])] * CV.DEG_TO_RAD
    hca_status = self.CCP.hca_status_values.get(pt_cp.vl["Lenkhilfe_2"]["LH2_Sta_HCA"])
    ret.steerFaultTemporary, ret.steerFaultPermanent = self.update_hca_state(hca_status)

    # Update gas, brakes, and gearshift.
    ret.gas = pt_cp.vl["Motor_3"]["Fahrpedal_Rohsignal"] / 100.0
    ret.gasPressed = ret.gas > 0
    ret.brake = pt_cp.vl["Bremse_5"]["Bremsdruck"] / 250.0  # FIXME: this is pressure in Bar, not sure what OP expects
    ret.brakePressed = bool(pt_cp.vl["Motor_2"]["Bremslichtschalter"])
    ret.parkingBrake = bool(pt_cp.vl["Kombi_1"]["Bremsinfo"])

    # Update gear and/or clutch position data.
    if self.CP.transmissionType == TransmissionType.automatic:
      ret.gearShifter = self.parse_gear_shifter(self.CCP.shifter_values.get(pt_cp.vl["Getriebe_1"]["Waehlhebelposition__Getriebe_1_"], None))
    elif self.CP.transmissionType == TransmissionType.manual:
      ret.clutchPressed = not pt_cp.vl["Motor_1"]["Kupplungsschalter"]
      reverse_light = bool(pt_cp.vl["Gate_Komf_1"]["GK1_Rueckfahr"])
      if reverse_light:
        ret.gearShifter = GearShifter.reverse
      else:
        ret.gearShifter = GearShifter.drive

    # Update door and trunk/hatch lid open status.
    ret.doorOpen = any([pt_cp.vl["Gate_Komf_1"]["GK1_Fa_Tuerkont"],
                        pt_cp.vl["Gate_Komf_1"]["BSK_BT_geoeffnet"],
                        pt_cp.vl["Gate_Komf_1"]["BSK_HL_geoeffnet"],
                        pt_cp.vl["Gate_Komf_1"]["BSK_HR_geoeffnet"],
                        pt_cp.vl["Gate_Komf_1"]["BSK_HD_Hauptraste"]])

    # Update seatbelt fastened status.
    ret.seatbeltUnlatched = not bool(pt_cp.vl["Airbag_1"]["Gurtschalter_Fahrer"])

    # Consume blind-spot monitoring info/warning LED states, if available.
    # Infostufe: BSM LED on, Warnung: BSM LED flashing
    if self.CP.enableBsm:
      ret.leftBlindspot = bool(ext_cp.vl["SWA_1"]["SWA_Infostufe_SWA_li"]) or bool(ext_cp.vl["SWA_1"]["SWA_Warnung_SWA_li"])
      ret.rightBlindspot = bool(ext_cp.vl["SWA_1"]["SWA_Infostufe_SWA_re"]) or bool(ext_cp.vl["SWA_1"]["SWA_Warnung_SWA_re"])

    # Consume factory LDW data relevant for factory SWA (Lane Change Assist)
    # and capture it for forwarding to the blind spot radar controller
    self.ldw_stock_values = cam_cp.vl["LDW_Status"] if self.CP.networkLocation == NetworkLocation.fwdCamera else {}

    # Stock FCW is considered active if the release bit for brake-jerk warning
    # is set. Stock AEB considered active if the partial braking or target
    # braking release bits are set.
    # Refer to VW Self Study Program 890253: Volkswagen Driver Assistance
    # Systems, chapters on Front Assist with Braking and City Emergency
    # Braking for the 2016 Passat NMS
    # TODO: deferred until we can collect data on pre-MY2016 behavior, AWV message may be shorter with fewer signals
    ret.stockFcw = False
    ret.stockAeb = False

    # Update ACC radar status.
    self.acc_type = ext_cp.vl["ACC_System"]["ACS_Typ_ACC"]
    ret.cruiseState.available = bool(pt_cp.vl["Motor_5"]["GRA_Hauptschalter"])
    ret.cruiseState.enabled = pt_cp.vl["Motor_2"]["GRA_Status"] in (1, 2)
    if self.CP.pcmCruise:
      ret.accFaulted = ext_cp.vl["ACC_GRA_Anzeige"]["ACA_StaACC"] in (6, 7)
    else:
      ret.accFaulted = pt_cp.vl["Motor_2"]["GRA_Status"] == 3

    # Update ACC setpoint. When the setpoint reads as 255, the driver has not
    # yet established an ACC setpoint, so treat it as zero.
    ret.cruiseState.speed = ext_cp.vl["ACC_GRA_Anzeige"]["ACA_V_Wunsch"] * CV.KPH_TO_MS
    if ret.cruiseState.speed > 70:  # 255 kph in m/s == no current setpoint
      ret.cruiseState.speed = 0

    # Update button states for turn signals and ACC controls, capture all ACC button state/config for passthrough
    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_stalk(300, pt_cp.vl["Gate_Komf_1"]["GK1_Blinker_li"],
                                                                            pt_cp.vl["Gate_Komf_1"]["GK1_Blinker_re"])
    ret.buttonEvents = self.create_button_events(pt_cp, self.CCP.BUTTONS)
    self.gra_stock_values = pt_cp.vl["GRA_Neu"]

    # Additional safety checks performed in CarInterface.
    ret.espDisabled = bool(pt_cp.vl["Bremse_1"]["ESP_Passiv_getastet"])

    self.frame += 1
    return ret
    
  def update_meb(self, pt_cp, cam_cp, ext_cp) -> structs.CarState:
    ret = structs.CarState()
    # Update vehicle speed and acceleration from ABS wheel speeds.
    ret.wheelSpeeds = self.get_wheel_speeds(
      pt_cp.vl["ESC_51"]["VL_Radgeschw"],
      pt_cp.vl["ESC_51"]["VR_Radgeschw"],
      pt_cp.vl["ESC_51"]["HL_Radgeschw"],
      pt_cp.vl["ESC_51"]["HR_Radgeschw"],
    )

    ret.vEgoRaw = float(np.mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr]))
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw == 0

    # Update EPS position and state info. For signed values, VW sends the sign in a separate signal.
    # LWI_01, MEP_EPS_01 steering angle differs from real steering angle (dynamic steering)
    ret.steeringAngleDeg = pt_cp.vl["LWI_01"]["LWI_Lenkradwinkel"] * (1, -1)[int(pt_cp.vl["LWI_01"]["LWI_VZ_Lenkradwinkel"])]
    ret.steeringRateDeg = pt_cp.vl["LWI_01"]["LWI_Lenkradw_Geschw"] * (1, -1)[int(pt_cp.vl["LWI_01"]["LWI_VZ_Lenkradw_Geschw"])]
    ret.steeringTorque = pt_cp.vl["LH_EPS_03"]["EPS_Lenkmoment"] * (1, -1)[int(pt_cp.vl["LH_EPS_03"]["EPS_VZ_Lenkmoment"])]
    ret.steeringPressed = abs(ret.steeringTorque) > self.CCP.STEER_DRIVER_ALLOWANCE
    
    ret.yawRate = pt_cp.vl["ESC_50"]["Yaw_Rate"] * (1, -1)[int(pt_cp.vl["ESC_50"]["Yaw_Rate_Sign"])] * CV.DEG_TO_RAD
    self.curvature = -pt_cp.vl["QFK_01"]["Curvature"] * (1, -1)[int(pt_cp.vl["QFK_01"]["Curvature_VZ"])]
    
    hca_status = self.CCP.hca_status_values.get(pt_cp.vl["QFK_01"]["LatCon_HCA_Status"])
    ret.steerFaultTemporary, ret.steerFaultPermanent = self.update_hca_state(hca_status)

    # VW Emergency Assist status tracking and mitigation
    self.eps_stock_values = pt_cp.vl["LH_EPS_03"]
    #ret.carFaultedNonCritical = cam_cp.vl["EA_01"]["EA_Funktionsstatus"] in (3, 4, 5, 6) # prepared, not tested

    # Update gas, brakes, and gearshift.
    ret.gasPressed = pt_cp.vl["Motor_54"]["Accelerator_Pressure"] > 0
    ret.gas = pt_cp.vl["Motor_54"]["Accelerator_Pressure"]
    ret.brakePressed = bool(pt_cp.vl["Motor_14"]["MO_Fahrer_bremst"]) # includes regen braking by user
    ret.brake = pt_cp.vl["ESC_51"]["Brake_Pressure"]
    ret.parkingBrake = pt_cp.vl["Gateway_73"]["EPB_Status"] in (1, 4) # EPB closing or closed
    # regen braking bool(pt_cp.vl["ESC_50"]['Regen_Braking']) TODO

    # Update gear and/or clutch position data.
    ret.gearShifter = self.parse_gear_shifter(self.CCP.shifter_values.get(pt_cp.vl["Getriebe_11"]["GE_Fahrstufe"], None))

    # Update door and trunk/hatch lid open status.
    ret.doorOpen = any([pt_cp.vl["ZV_02"]["ZV_FT_offen"],
                        pt_cp.vl["ZV_02"]["ZV_BT_offen"],
                        pt_cp.vl["ZV_02"]["ZV_HFS_offen"],
                        pt_cp.vl["ZV_02"]["ZV_HBFS_offen"],
                        pt_cp.vl["ZV_02"]["ZV_HD_offen"]])

    # Update seatbelt fastened status.
    ret.seatbeltUnlatched = pt_cp.vl["Airbag_02"]["AB_Gurtschloss_FA"] != 3

    # Consume blind-spot monitoring info/warning LED states, if available.
    # Infostufe: BSM LED on, Warnung: BSM LED flashing
    if self.CP.enableBsm:
      ret.leftBlindspot = ext_cp.vl["MEB_Side_Assist_01"]["Blind_Spot_Left"] > 0
      ret.rightBlindspot = ext_cp.vl["MEB_Side_Assist_01"]["Blind_Spot_Right"] > 0

    # Consume factory LDW data relevant for factory SWA (Lane Change Assist)
    # and capture it for forwarding to the blind spot radar controller
    self.ldw_stock_values = cam_cp.vl["LDW_02"]

    ret.stockFcw = bool(pt_cp.vl["VMM_02"]["FCW_Active"]) or bool(ext_cp.vl["AWV_03"]["FCW_Active"])
    ret.stockAeb = bool(pt_cp.vl["VMM_02"]["AEB_Active"])

    self.acc_type = ext_cp.vl["ACC_18"]["ACC_Typ"]
    self.travel_assist_available = bool(cam_cp.vl["TA_01"]["Travel_Assist_Available"])

    ret.cruiseState.available = pt_cp.vl["Motor_51"]["TSK_Status"] in (2, 3, 4, 5)
    ret.cruiseState.enabled   = pt_cp.vl["Motor_51"]["TSK_Status"] in (3, 4, 5)

    if self.CP.pcmCruise:
      # Cruise Control mode; check for distance UI setting from the radar.
      # ECM does not manage this, so do not need to check for openpilot longitudinal
      ret.cruiseState.nonAdaptive = bool(ext_cp.vl["MEB_ACC_01"]["ACC_Limiter_Mode"])
    else:
      # Speed limiter mode; ECM faults if we command ACC while not pcmCruise
      ret.cruiseState.nonAdaptive = bool(pt_cp.vl["Motor_51"]["TSK_Limiter_ausgewaehlt"])

    ret.accFaulted = pt_cp.vl["Motor_51"]["TSK_Status"] in (6, 7)

    self.esp_hold_confirmation = bool(pt_cp.vl["VMM_02"]["ESP_Hold"])
    ret.cruiseState.standstill = self.CP.pcmCruise and self.esp_hold_confirmation

    # Update ACC setpoint. When the setpoint is zero or there's an error, the
    # radar sends a set-speed of ~90.69 m/s / 203mph.
    if self.CP.pcmCruise:
      ret.cruiseState.speed = int(round(ext_cp.vl["MEB_ACC_01"]["ACC_Wunschgeschw_02"])) * CV.KPH_TO_MS
      if ret.cruiseState.speed > 90:
        ret.cruiseState.speed = 0

    # Update button states for turn signals and ACC controls, capture all ACC button state/config for passthrough
    ret.leftBlinker = bool(pt_cp.vl["Blinkmodi_02"]["BM_links"])
    ret.rightBlinker = bool(pt_cp.vl["Blinkmodi_02"]["BM_rechts"])
    ret.buttonEvents = self.create_button_events(pt_cp, self.CCP.BUTTONS)
    self.gra_stock_values = pt_cp.vl["GRA_ACC_01"]

    # Additional safety checks performed in CarInterface.
    ret.espDisabled = bool(pt_cp.vl["ESP_21"]["ESP_Tastung_passiv"]) # this is also true for ESC Sport mode
    ret.espActive = bool(pt_cp.vl["ESP_21"]["ESP_Eingriff"])

    # EV battery charge WattHours
    ret.fuelGauge = pt_cp.vl["Motor_16"]["MO_Energieinhalt_BMS"]

    self.frame += 1
    return ret

  def update_hca_state(self, hca_status, drive_mode=True):
    # Treat FAULT as temporary for worst likely EPS recovery time, for cars without factory Lane Assist
    # DISABLED means the EPS hasn't been configured to support Lane Assist
    self.eps_init_complete = self.eps_init_complete or (hca_status in ("DISABLED", "READY", "ACTIVE") or self.frame > 600)
    perm_fault = drive_mode and hca_status == "DISABLED" or (self.eps_init_complete and hca_status == "FAULT")
    temp_fault = drive_mode and hca_status in ("REJECTED", "PREEMPTED") or not self.eps_init_complete
    return temp_fault, perm_fault

  @staticmethod
  def get_can_parsers(CP):
    if CP.flags & VolkswagenFlags.PQ:
      return CarState.get_can_parsers_pq(CP)
    elif CP.flags & VolkswagenFlags.MEB:
      return CarState.get_can_parsers_meb(CP)

    pt_messages = [
      # sig_address, frequency
      ("LWI_01", 100),      # From J500 Steering Assist with integrated sensors
      ("LH_EPS_03", 100),   # From J500 Steering Assist with integrated sensors
      ("ESP_19", 100),      # From J104 ABS/ESP controller
      ("ESP_05", 50),       # From J104 ABS/ESP controller
      ("ESP_21", 50),       # From J104 ABS/ESP controller
      ("Motor_20", 50),     # From J623 Engine control module
      ("TSK_06", 50),       # From J623 Engine control module
      ("ESP_02", 50),       # From J104 ABS/ESP controller
      ("GRA_ACC_01", 33),   # From J533 CAN gateway (via LIN from steering wheel controls)
      ("Gateway_73", 20),   # From J533 CAN gateway (aggregated data)
      ("Gateway_72", 10),   # From J533 CAN gateway (aggregated data)
      ("Motor_14", 10),     # From J623 Engine control module
      ("Airbag_02", 5),     # From J234 Airbag control module
      ("Kombi_01", 2),      # From J285 Instrument cluster
      ("Blinkmodi_02", 1),  # From J519 BCM (sent at 1Hz when no lights active, 50Hz when active)
      ("Kombi_03", 0),      # From J285 instrument cluster (not present on older cars, 1Hz when present)
    ]

    if CP.transmissionType == TransmissionType.direct:
      pt_messages.append(("Motor_EV_01", 10))  # From J??? unknown EV control module

    if CP.networkLocation == NetworkLocation.fwdCamera:
      # Radars are here on CANBUS.pt
      pt_messages += MqbExtraSignals.fwd_radar_messages
      if CP.enableBsm:
        pt_messages += MqbExtraSignals.bsm_radar_messages

    cam_messages = []
    if CP.flags & VolkswagenFlags.STOCK_HCA_PRESENT:
      cam_messages += [
        ("HCA_01", 1),  # From R242 Driver assistance camera, 50Hz if steering/1Hz if not
      ]

    if CP.networkLocation == NetworkLocation.fwdCamera:
      cam_messages += [
        # sig_address, frequency
        ("LDW_02", 10)      # From R242 Driver assistance camera
      ]
    else:
      # Radars are here on CANBUS.cam
      cam_messages += MqbExtraSignals.fwd_radar_messages
      if CP.enableBsm:
        cam_messages += MqbExtraSignals.bsm_radar_messages

    return {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], pt_messages, CANBUS.pt),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], cam_messages, CANBUS.cam),
    }

  @staticmethod
  def get_can_parsers_pq(CP):
    pt_messages = [
      # sig_address, frequency
      ("Bremse_1", 100),    # From J104 ABS/ESP controller
      ("Bremse_3", 100),    # From J104 ABS/ESP controller
      ("Lenkhilfe_3", 100),  # From J500 Steering Assist with integrated sensors
      ("Lenkwinkel_1", 100),  # From J500 Steering Assist with integrated sensors
      ("Motor_3", 100),     # From J623 Engine control module
      ("Airbag_1", 50),     # From J234 Airbag control module
      ("Bremse_5", 50),     # From J104 ABS/ESP controller
      ("GRA_Neu", 50),      # From J??? steering wheel control buttons
      ("Kombi_1", 50),      # From J285 Instrument cluster
      ("Motor_2", 50),      # From J623 Engine control module
      ("Motor_5", 50),      # From J623 Engine control module
      ("Lenkhilfe_2", 20),  # From J500 Steering Assist with integrated sensors
      ("Gate_Komf_1", 10),  # From J533 CAN gateway
    ]

    if CP.transmissionType == TransmissionType.automatic:
      pt_messages += [("Getriebe_1", 100)]  # From J743 Auto transmission control module
    elif CP.transmissionType == TransmissionType.manual:
      pt_messages += [("Motor_1", 100)]  # From J623 Engine control module

    if CP.networkLocation == NetworkLocation.fwdCamera:
      # Extended CAN devices other than the camera are here on CANBUS.pt
      pt_messages += PqExtraSignals.fwd_radar_messages
      if CP.enableBsm:
        pt_messages += PqExtraSignals.bsm_radar_messages

    cam_messages = []
    if CP.networkLocation == NetworkLocation.fwdCamera:
      cam_messages += [
        # sig_address, frequency
        ("LDW_Status", 10)      # From R242 Driver assistance camera
      ]

    if CP.networkLocation == NetworkLocation.gateway:
      # Radars are here on CANBUS.cam
      cam_messages += PqExtraSignals.fwd_radar_messages
      if CP.enableBsm:
        cam_messages += PqExtraSignals.bsm_radar_messages

    return {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], pt_messages, CANBUS.pt),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], cam_messages, CANBUS.cam),
    }
    
  @staticmethod
  def get_can_parsers_meb(CP):
    pt_messages = [
      # sig_address, frequency
      ("LWI_01", 100),            # From J500 Steering Assist with integrated sensors
      ("GRA_ACC_01", 33),         # From J533 CAN gateway (via LIN from steering wheel controls)
      ("Airbag_02", 5),           # From J234 Airbag control module
      ("Motor_14", 10),           # From J623 Engine control module
      ("Motor_16", 2),            # From J623 Engine control module
      ("Blinkmodi_02", 2),        # From J519 BCM (sent at 1Hz when no lights active, 50Hz when active)
      ("LH_EPS_03", 100),         # From J500 Steering Assist with integrated sensors
      ("Getriebe_11", 100),       # From J743 Auto transmission control module
      ("ZV_02", 5),               # From ZV
      ("QFK_01", 100),            # From Steering
      ("ESP_21", 50),             #
      ("EML_06", 50),             #
      ("ESC_51", 100),            #
      ("Motor_54", 10),           #
      ("ESC_50", 50),             #
      ("VMM_02", 50),             #
      ("Gateway_73", 20),         #
      ("SAM_01", 5),              #
      ("Motor_51", 50),           #
    ]

    if CP.networkLocation == NetworkLocation.fwdCamera:
      # Radars are here on CANBUS.pt
      pt_messages += MebExtraSignals.fwd_radar_messages
      if CP.enableBsm:
        pt_messages += MebExtraSignals.bsm_radar_messages

    cam_messages = [
      # sig_address, frequency
      ("LDW_02", 10),     # From R242 Driver assistance camera
      ("TA_01", 10),      # From R242 Driver assistance camera (Travel Assist)
    ]
    
    if CP.networkLocation == NetworkLocation.gateway:
      # Radars are here on CANBUS.cam
      cam_messages += MebExtraSignals.fwd_radar_messages
      if CP.enableBsm:
        cam_messages += MebExtraSignals.bsm_radar_messages

    return {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], pt_messages, CANBUS.pt),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], cam_messages, CANBUS.cam),
    }


class MqbExtraSignals:
  # Additional signal and message lists for optional or bus-portable controllers
  fwd_radar_messages = [
    ("ACC_06", 50),                              # From J428 ACC radar control module
    ("ACC_10", 50),                              # From J428 ACC radar control module
    ("ACC_02", 17),                              # From J428 ACC radar control module
  ]
  bsm_radar_messages = [
    ("SWA_01", 20),                              # From J1086 Lane Change Assist
  ]


class PqExtraSignals:
  # Additional signal and message lists for optional or bus-portable controllers
  fwd_radar_messages = [
    ("ACC_System", 50),                          # From J428 ACC radar control module
    ("ACC_GRA_Anzeige", 25),                     # From J428 ACC radar control module
  ]
  bsm_radar_messages = [
    ("SWA_1", 20),                               # From J1086 Lane Change Assist
  ]


class MebExtraSignals:
  # Additional signal and message lists for optional or bus-portable controllers
  fwd_radar_messages = [
    ("MEB_ACC_01", 17),        #
    ("ACC_18", 50),            #
    ("AWV_03", 1),             # Front Collision Detection (1 Hz when inactive, 50 Hz when active)
    #("MEB_Distance_01", 25),  #
  ]
  bsm_radar_messages = [
    ("MEB_Side_Assist_01", 20),
  ]
