import collections
from collections import abc
collections.MutableMapping = abc.MutableMapping

from dronekit import connect, VehicleMode
from pymavlink import mavutil
from time import sleep
from math import radians




def change_mode(vehicle, flight_mode: str):
    vehicle.mode = VehicleMode(flight_mode)
    # Wait until mode changed
    vehicle.wait_for_mode(flight_mode)

    if vehicle.mode.name == flight_mode:
        print("Mode changed to", flight_mode)
    else:
        print("Failed to Change Mode")


def arm(vehicle):
    print("Arming motors")
    vehicle.arm(wait=True) # Wait until armed
    print("Vehicle Armed")


def disarm(vehicle):
    print("Disarming motors")
    vehicle.disarm(wait=True) # Wait until disarmed
    print("Vehicle Disarmed")


def send_ned_velocity(vehicle, velocity_x, velocity_y, velocity_z):
    to_send = vehicle.message_factory.set_position_target_local_ned_encode(
                    10, # time_boot_ms
                    0, # DroneKit will automatically update the value with the correct ID for the connected vehicle
                    0, # Not updated by DroneKit, but should be set to 0 (broadcast) unless the message is really intended for a specific component
                    mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # Frame
                    0b0000011111000111,  # Typemask
                    0, 0, 0,  # XYZ Position (m)
                    velocity_x, velocity_y, velocity_z,
                    # XYZ Velocity (m/s)
                    0, 0, 0,  # XYZ Acceleration (m/s/s)
                    0,  # Yaw setpoint (rad)
                    0  # Yaw rate (rad/s)
                    )
    # Send command to vehicle
    vehicle.send_mavlink(to_send)


def send_global_position(vehicle, code):
    if global_variable.indoor_left == True:
        latitude = global_variable.waypoint_item_left[code]["latitude"]
        longitude = global_variable.waypoint_item_left[code]["longitude"]
        altitude = global_variable.waypoint_item_left[code]["altitude"]
    else:
        latitude = global_variable.waypoint_item_right[code]["latitude"]
        longitude = global_variable.waypoint_item_right[code]["longitude"]
        altitude = global_variable.waypoint_item_right[code]["altitude"]

    to_send = vehicle.message_factory.set_position_target_global_int_encode(10, 0, 0, 
                              mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, # Frame
                              0b110111111000,  # Typemask
                              int(latitude * 10**7), 
                              int(longitude * 10**7), 
                              altitude,
                              0, 0, 0, # Velocity
                              0, 0, 0, # Acceleration
                              0, 0) # Yaw
    vehicle.send_mavlink(to_send)


def override_global_position(vehicle):
    to_send = vehicle.message_factory.command_long_encode(0, 0, 
                              mavutil.mavlink.MAV_CMD_OVERRIDE_GOTO, 
                              0, 
                              mavutil.mavlink.MAV_GOTO_DO_HOLD, 
                              mavutil.mavlink.MAV_GOTO_HOLD_AT_CURRENT_POSITION, 
                              0, 0, 0, 0, 0) 
    vehicle.send_mavlink(to_send)


def condition_yaw(vehicle, target_heading, direction, yaw_speed=30):
    to_send = vehicle.message_factory.command_long_encode(0, 0,
                                  mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                                  0,  # Confirmation
                                  target_heading,  # Yaw in (deg)
                                  yaw_speed,  # Yaw speed (deg/s)
                                  direction,  # -1=ccw, 1=cw
                                  1,  # 0=absolute angle, 1=relative angle
                                  0, 0, 0)
    # Send command to vehicle
    vehicle.send_mavlink(to_send)
    
    # Wait to ensure yaw is completed
    sleep(target_heading / yaw_speed)



def takeoff(vehicle: object, target_altitude: float) -> object:
    change_mode(vehicle, "GUIDED")
    arm(vehicle)

    while not vehicle.armed:      
        print("Waiting for arming...")
        sleep(0.25)

    print("Taking off..")
    vehicle.simple_takeoff(target_altitude)


def cek_altitude(vehicle, target_altitude):
    altitude_counter = 0
    while True:
        try:
            current_altitude = global_variable.rngfnd["25"]["current_distance"] / 100
        except KeyError:
            current_altitude = target_altitude - 0.2

        print(f"Altitude: {current_altitude} m of {target_altitude} m")      

        if vehicle.system_status == "ACTIVE":
            if current_altitude < 0.95 * target_altitude: 
                send_ned_velocity(vehicle, 0, 0, -0.25)
                print("Naik to reach altitude")
            
            if current_altitude > 1.1 * target_altitude: 
                send_ned_velocity(vehicle, 0, 0, 0.3)
                print("Turun to reach altitude")

            if 0.95 * target_altitude <= current_altitude <= 1.1 * target_altitude:
                altitude_counter += 1
                print(f"ALT: {altitude_counter}")
                # Stop if altitude reached for 3 counts
                if altitude_counter >= 3:
                    print("Target Altitude Reached")
                    break


def land(vehicle):
    print("Landing..")
    change_mode(vehicle, "LAND")

    while True:
        try:
            current_altitude = global_variable.rngfnd["25"]["current_distance"] / 100
        except KeyError:
            current_altitude = 0.2

        current_status = vehicle.system_status
        print(f"Altitude: {current_altitude} m of 0 m")

        if current_status == "STANDBY":
            print("Vehicle landed")
            break




