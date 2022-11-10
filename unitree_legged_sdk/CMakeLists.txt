cmake_minimum_required(VERSION 2.8.3)
project(unitree_legged_sdk)

# add_compile_options(-std=c++11)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    geometry_msgs
    std_msgs
)

catkin_package(INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs
)

include_directories(
    include
    ${Boost_INCLUDE_DIR}
    ${catkin_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "-O3")
set(EXTRA_LIBS -pthread libunitree_legged_sdk.so lcm)
link_directories(lib)

# change into your Dir
add_executable(example_walk src/example_walk.cpp)
target_link_libraries(example_walk ${EXTRA_LIBS} ${catkin_LIBRARIES})
# add_dependencies(example_walk ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(unitree_planning src/unitree_planning.cpp)
target_link_libraries(unitree_planning ${EXTRA_LIBS} ${catkin_LIBRARIES})

add_executable(speed_calibration src/speed_calibration.cpp)
target_link_libraries(speed_calibration ${EXTRA_LIBS} ${catkin_LIBRARIES})

add_executable(control_via_keyboard src/control_via_keyboard.cpp)
target_link_libraries(control_via_keyboard ${EXTRA_LIBS} ${catkin_LIBRARIES})
add_dependencies(control_via_keyboard ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# # one pc one process
# add_executable(example_position examples/example_position.cpp)
# target_link_libraries(example_position ${EXTRA_LIBS})

# add_executable(example_velocity examples/example_velocity.cpp)
# target_link_libraries(example_velocity ${EXTRA_LIBS})

# add_executable(example_torque examples/example_torque.cpp)
# target_link_libraries(example_torque ${EXTRA_LIBS})

# add_executable(example_walk examples/example_walk.cpp)
# target_link_libraries(example_walk ${EXTRA_LIBS})

# # one pc multi process
# add_executable(lcm_server_high examples/lcm_server_high.cpp)
# target_link_libraries(lcm_server_high ${EXTRA_LIBS})

# add_executable(lcm_server_low examples/lcm_server_low.cpp)
# target_link_libraries(lcm_server_low ${EXTRA_LIBS})

# # multi pc
# add_executable(udp_send_test examples/multi_pc_udp_send.cpp)
# target_link_libraries(udp_send_test ${EXTRA_LIBS})

# add_executable(udp_recv_test examples/multi_pc_udp_recv.cpp)
# target_link_libraries(udp_recv_test ${EXTRA_LIBS})