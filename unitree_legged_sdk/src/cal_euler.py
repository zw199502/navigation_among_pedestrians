from math import asin, atan2
# q_0 = -0.905#w
# q_1 = 0.0#x
# q_2 = 0.427#y
# q_3 = 0.0#z
q_0 = 0.9307691#w
q_1 = -0.00445106#x
q_2 = -0.00172516#y
q_3 = -0.3655763#z
roll = atan2(2.0 * (q_0 * q_1 + q_2 * q_3), 1.0 - 2.0 * (q_1 * q_1 + q_2 * q_2))
pitch = asin(2.0 * (q_0 * q_2 - q_1 * q_3))
yaw = atan2(2.0 * (q_0 * q_3 + q_1 * q_2), 1.0 - 2.0 * (q_2 * q_2 + q_3 * q_3))
print(roll, pitch, yaw)

#pitch, anti-clockwise, negative
#pitch, clockwise, positive