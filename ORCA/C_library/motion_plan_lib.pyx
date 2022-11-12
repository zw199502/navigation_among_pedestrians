## python setup.py build_ext --inplace ####
from libc.stdlib cimport malloc, free

cdef extern from "math.h":
    float cos(float theta)
    float sin(float theta)
    float sqrt(float item)
    float atan2(float y, float x)
    float hypot(float dx, float dy)
    float fabs(float item)

cdef int num_line
cdef int num_circle
cdef int num_scan
cdef float *lines
cdef float *circles
cdef float *scans
cdef float *scan_lines
cdef float *robot_pose
cdef float laser_resolution
cdef float PI = 3.141592654

# transform previous lasers into current robot frame
cdef float *scan_last_1
cdef float *scan_last_2
cdef float *scan_last_3
cdef float *scan_end_last_1
cdef float *scan_end_last_2
cdef float *scan_end_last_3

def InitializeEnv(_num_line: int, _num_circle: int, _num_scan: int, _laser_resolution: float):
    global num_line, num_circle, num_scan
    global lines, circles, scans, scan_lines, robot_pose
    global laser_resolution
    global scan_last_1, scan_last_2, scan_last_3
    global scan_end_last_1, scan_end_last_2, scan_end_last_3

    num_line = _num_line
    num_circle = _num_circle
    num_scan = _num_scan
    laser_resolution = _laser_resolution

    lines = <float*>malloc(num_line * 4 * sizeof(float))
    circles = <float*>malloc(num_circle * 3 * sizeof(float))
    scans = <float*>malloc(num_scan * sizeof(float))
    scan_lines = <float*>malloc(num_scan * 4 * sizeof(float))
    robot_pose = <float*>malloc(3 * sizeof(float))

    scan_last_1 = <float*>malloc(num_scan * sizeof(float))
    scan_last_2 = <float*>malloc(num_scan * sizeof(float))
    scan_last_3 = <float*>malloc(num_scan * sizeof(float))
    for i in range(num_scan):
        scan_last_1[i] = 999.9
        scan_last_2[i] = 999.9
        scan_last_3[i] = 999.9
    scan_end_last_1 = <float*>malloc(num_scan * 2 * sizeof(float))
    scan_end_last_2 = <float*>malloc(num_scan * 2 * sizeof(float))
    scan_end_last_3 = <float*>malloc(num_scan * 2 * sizeof(float))

def set_lines(index: int, item: float):
    global lines
    lines[index] = item

def set_circles(index: int, item: float):
    global circles
    circles[index] = item

def set_robot_pose(x: float, y: float, yaw: float):
    global robot_pose
    robot_pose[0] = x
    robot_pose[1] = y
    robot_pose[2] = yaw

def set_scan_end_last(x: float, y: float, index: int, scan: int):
    global scan_end_last_1, scan_end_last_2, scan_end_last_3
    if scan == 1:
        scan_end_last_1[2 * index] = x
        scan_end_last_1[2 * index + 1] = y
    elif scan == 2:
        scan_end_last_2[2 * index] = x
        scan_end_last_2[2 * index + 1] = y
    else:
        scan_end_last_3[2 * index] = x
        scan_end_last_3[2 * index + 1] = y


def cal_laser():
    global num_line, num_circle, num_scan, lines, circles, scans, scan_lines, robot_pose, laser_resolution
    cdef float _intersection_x = 0.
    cdef float _intersection_y = 0.
    cdef float scan_range = 0.
    cdef float min_range = 999.9
    cdef float float_num_scan = num_scan
    cdef float float_i = 0.
    cdef float angle_rel = 0.
    cdef float angle_abs = 0.
    cdef float line_unit_vector[2]
    cdef float a1, b1, c1, x0, y0, x1, y1, f0, f1, dx, dy, a2, b2, c2, intersection_x, intersection_y
    cdef float line_vector[2]
    cdef float x2, y2, r, f2, d, a3, b3, c3, intermediate_x, intermediate_y
    cdef float temp, l_vector, intersection_x_1, intersection_x_2, intersection_y_1, intersection_y_2
    cdef: 
        float *line_vector_1 
        float *line_vector_2
    line_vector_1 = <float*>malloc(2 * sizeof(float))
    line_vector_2 = <float*>malloc(2 * sizeof(float))
    for i in range(num_scan):
        min_range = 999.9
        float_i = i
        angle_rel = (float_i - (float_num_scan - 1.0) / 2.0) * laser_resolution
        # laser angle range: [-pi, pi]
        if angle_rel < -PI:
            angle_rel = -PI
        if angle_rel > PI:
            angle_rel = PI
        angle_abs = angle_rel + robot_pose[2]
        line_unit_vector[:] = [cos(angle_abs), sin(angle_abs)]
        # a1,b1,c1 are the parameters of line
        a1 = line_unit_vector[1]
        b1 = -line_unit_vector[0]
        c1 = robot_pose[1] * line_unit_vector[0] - robot_pose[0] * line_unit_vector[1]

        for j in range(num_line):
            x0 = lines[j * 4 + 0]
            y0 = lines[j * 4 + 1]
            x1 = lines[j * 4 + 2]
            y1 = lines[j * 4 + 3]
            f0 = a1 * x0 + b1 * y0 + c1
            f1 = a1 * x1 + b1 * y1 + c1
            if f0 * f1 > 0: # the two points are located on the same side
                continue
            else:
                dx = x1 - x0
                dy = y1 - y0
                a2 = dy
                b2 = -dx
                c2 = y0 * dx - x0 * dy
                intersection_x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
                intersection_y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
                # intersection is always in front of the robot
                line_vector[:] = [intersection_x - robot_pose[0], intersection_y - robot_pose[1]]
                # the intersection point must be located in the one direction of the line
                if (line_vector[0] * line_unit_vector[0] > 0) or (line_vector[1] * line_unit_vector[1] > 0):
                    scan_range = hypot(intersection_x - robot_pose[0], intersection_y - robot_pose[1])
                    if scan_range < min_range:
                        _intersection_x = intersection_x
                        _intersection_y = intersection_y
                        min_range = scan_range

        for k in range(num_circle):
            x2 = circles[3 * k + 0]
            y2 = circles[3 * k + 1]
            r = circles[3 * k + 2]
            f2 = a1 * x2 + b1 * y2 + c1
            
            d = fabs(f2) / hypot(a1, b1)
            if d > r:
                continue
            else:
                a3 = b1
                b3 = - a1
                c3 = -(a3 * x2 + b3 * y2)
                intermediate_x = (b1 * c3 - b3 * c1) / (a1 * b3 - a3 * b1)
                intermediate_y = (a3 * c1 - a1 * c3) / (a1 * b3 - a3 * b1)
                if d == r:
                    _intersection_x = intermediate_x
                    _intersection_y = intermediate_y
                else:
                    if d >= 0 and d < r:
                        temp = r * r - d * d
                        if temp < 0.0 or temp > 999.0:
                            continue
                        l_vector = sqrt(temp)
                        intersection_x_1 = l_vector * line_unit_vector[0] + intermediate_x
                        intersection_y_1 = l_vector * line_unit_vector[1] + intermediate_y
                        intersection_x_2 = -l_vector * line_unit_vector[0] + intermediate_x
                        intersection_y_2 = -l_vector * line_unit_vector[1] + intermediate_y
                        line_vector_1[0] = intersection_x_1 - robot_pose[0]
                        line_vector_1[1] = intersection_y_1 - robot_pose[1]
                        # the intersection point must be located in the one direction of the line
                        if (line_vector_1[0] * line_unit_vector[0]) > 0 or (line_vector_1[1] * line_unit_vector[1] > 0):
                            scan_range = hypot(intersection_x_1 - robot_pose[0], intersection_y_1 - robot_pose[1])
                            if scan_range < min_range:
                                min_range = scan_range
                                _intersection_x = intersection_x_1
                                _intersection_y = intersection_y_1
                        line_vector_2[0] = intersection_x_2 - robot_pose[0]
                        line_vector_2[1] = intersection_y_2 - robot_pose[1]
                        if (line_vector_2[0] * line_unit_vector[0]) > 0 or (line_vector_2[1] * line_unit_vector[1] > 0):
                            scan_range = hypot(intersection_x_2 - robot_pose[0], intersection_y_2 - robot_pose[1])
                            if scan_range < min_range:
                                min_range = scan_range
                                _intersection_x = intersection_x_2
                                _intersection_y = intersection_y_2

        scans[i] = min_range
        scan_lines[4 * i + 0] = robot_pose[0]
        scan_lines[4 * i + 1] = robot_pose[1]
        scan_lines[4 * i + 2] = _intersection_x
        scan_lines[4 * i + 3] = _intersection_y
    free(line_vector_1)
    free(line_vector_2)

def transform_scan_last():
    global robot_pose, num_scan, laser_resolution
    global scan_last_1, scan_last_2, scan_last_3
    global scan_end_last_1, scan_end_last_2, scan_end_last_3
    cdef float dx, dy, theta, scan_x, scan_y, scan_angle
    cdef int scan_index
    theta = robot_pose[2]
    for i in range(num_scan):
        dx = scan_end_last_1[2 * i] - robot_pose[0]
        dy = scan_end_last_1[2 * i + 1] - robot_pose[1]
        scan_x = dy * sin(theta) + dx * cos(theta)
        scan_y = dy * cos(theta) - dx * sin(theta)
        scan_angle = atan2(scan_y, scan_x)
        if scan_angle >= -PI and scan_angle <= PI:
            scan_index = int((scan_angle + (PI - laser_resolution / 2.0)) / laser_resolution)
            if scan_index >=0 and scan_index < num_scan:
                scan_last_1[scan_index] = hypot(scan_x, scan_y)
        
        dx = scan_end_last_2[2 * i] - robot_pose[0]
        dy = scan_end_last_2[2 * i + 1] - robot_pose[1]
        scan_x = dy * sin(theta) + dx * cos(theta)
        scan_y = dy * cos(theta) - dx * sin(theta)
        scan_angle = atan2(scan_y, scan_x)
        if scan_angle >= -PI and scan_angle <= PI:
            scan_index = int((scan_angle + (PI - laser_resolution / 2.0)) / laser_resolution)
            if scan_index >=0 and scan_index < num_scan:
                scan_last_2[scan_index] = hypot(scan_x, scan_y)

        dx = scan_end_last_3[2 * i] - robot_pose[0]
        dy = scan_end_last_3[2 * i + 1] - robot_pose[1]
        scan_x = dy * sin(theta) + dx * cos(theta)
        scan_y = dy * cos(theta) - dx * sin(theta)
        scan_angle = atan2(scan_y, scan_x)
        if scan_angle >= -PI and scan_angle <= PI:
            scan_index = int((scan_angle + (PI - laser_resolution / 2.0)) / laser_resolution)
            if scan_index >=0 and scan_index < num_scan:
                scan_last_3[scan_index] = hypot(scan_x, scan_y)    

    # for i in range(num_scan):
    #     if i == 0:
    #         if scan_last_1[i] > 999:
    #             scan_last_1[i] = min([scan_last_1[num_scan - 1], scan_last_1[num_scan - 2],\
    #                 scan_last_1[1], scan_last_1[2]])   
    #         if scan_last_2[i] > 999:
    #             scan_last_2[i] = min([scan_last_2[num_scan - 1], scan_last_2[num_scan - 2],\
    #                 scan_last_2[1], scan_last_2[2]]) 
    #         if scan_last_3[i] > 999:
    #             scan_last_3[i] = min([scan_last_3[num_scan - 1], scan_last_3[num_scan - 2],\
    #                 scan_last_3[1], scan_last_3[2]])
    #     elif i == 1:
    #         if scan_last_1[i] > 999:
    #             scan_last_1[i] = min([scan_last_1[num_scan - 1], scan_last_1[0],\
    #                 scan_last_1[3], scan_last_1[2]])   
    #         if scan_last_2[i] > 999:
    #             scan_last_2[i] = min([scan_last_2[num_scan - 1], scan_last_2[0],\
    #                 scan_last_2[3], scan_last_2[2]]) 
    #         if scan_last_3[i] > 999:
    #             scan_last_3[i] = min([scan_last_3[num_scan - 1], scan_last_3[0],\
    #                 scan_last_3[3], scan_last_3[2]])
        
    #     else:
    #         if scan_last_1[i] > 999:
    #             scan_last_1[i] = min([scan_last_1[i - 1], scan_last_1[i - 2], \
    #                 scan_last_1[(i + 1) % num_scan], scan_last_1[(i + 2) % num_scan]])   
    #         if scan_last_2[i] > 999:
    #             scan_last_2[i] = min([scan_last_2[i - 1], scan_last_2[i - 2], \
    #                 scan_last_2[(i + 1) % num_scan], scan_last_2[(i + 2) % num_scan]])   
    #         if scan_last_3[i] > 999:
    #             scan_last_3[i] = min([scan_last_3[i - 1], scan_last_3[i - 2], \
    #                 scan_last_3[(i + 1) % num_scan], scan_last_3[(i + 2) % num_scan]])   

def get_scan(index: int):
    global scans
    return scans[index]

def get_scan_line(index: int):
    global scan_lines
    return scan_lines[index]

def get_last_scan(index: int, scan: int):
    global scan_last_1, scan_last_2, scan_last_3
    if scan == 1:
        return scan_last_1[index]
    elif scan == 2:
        return scan_last_2[index]
    else:
        return scan_last_3[index]

def ReleaseEnv():
    global lines, circles, scans, scan_lines, robot_pose
    global scan_last_1, scan_last_2, scan_last_3
    global scan_end_last_1, scan_end_last_2, scan_end_last_3
    free(lines)
    free(circles)
    free(scans)
    free(scan_lines)
    free(robot_pose)
    free(scan_last_1)
    free(scan_last_2)
    free(scan_last_3)
    free(scan_end_last_1)
    free(scan_end_last_2)
    free(scan_end_last_3)