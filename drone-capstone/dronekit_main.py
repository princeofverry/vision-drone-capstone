
from dronekit_control import *
import config


def init_vehicle():
    vehicle = init_connection(config.controller_address)
    # # Check the arming status
    # if vehicle.is_armable == True:
    #     print("Flight controller is armable.")
    # else:
    #     print("Flight controller is not armable.")
    #     vehicle.wait_for_armable()

    return vehicle        
        

def init_connection(address):
    print("Bismillahirrohmanirrahim...")
    vehicle = connect(address, wait_ready=True)

    if not vehicle:
        print("Connection failed, please check the connection")
        exit(1)
    return vehicle


def main():
    print('Main Program')

    # init all components
    vehicle = init_vehicle()
    
    sleep(2)


    input("Press enter to start flying after tracking program are loaded..")

    #arm(vehicle)

    # # Takeoff
    takeoff(vehicle, 0.55)
    sleep(10)
    change_mode(vehicle, "LAND")

    # # Cek Altitude
    # cek_altitude(vehicle, 0.55)
    # sleep(1)


    # for _ in range(4):
    #     send_ned_velocity(vehicle, 0.5, 0, 0)




main()