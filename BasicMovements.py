from djitellopy import tello

from time import sleep

me = tello.Tello()

# import logging
# me.LOGGER.setLevel(logging.DEBUG)

me.connect()

# sleep(2) 

print(me.get_battery())

# me.takeoff()

# me.send_rc_control(0, 50, 0, 0)

# sleep(2) 

# me.send_rc_control(0, 0, 0, 30)

# sleep(2)

# me.send_rc_control(0, 0, 0, 0)

# me.land()