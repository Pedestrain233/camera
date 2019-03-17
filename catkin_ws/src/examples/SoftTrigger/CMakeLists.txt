cmake_minimum_required(VERSION 2.8.3)
project(tiscamerasample C CXX)

find_package(catkin REQUIRED COMPONENTS
   roscpp
   rospy
   std_msgs
   message_generation
)

include_directories(
#${PROJET_SOURCE_DIR}
#${catkin_INCLUDE_DIRS}
#${tiscamera_INCLUDE_DIRS} 
${tiscamera_include}
)

link_directories(${PROJECT_SOURCE_DIR})
include_directories(/opt/ros/melodic/include/)

#set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
#set(CMAKE_INCLUDE_PATH "include_path")
#set(CMAKE_LIBRARY_PATH "lib_path")





link_libraries(tiscamera)

set(tiscamera_clude
    include/
    ${catkin_INCLUDE_DIRS})

catkin_package(
    INCLUDE_DIRS ${tiscamera_include}
    LIBRARIES ${tiscamera}
)

add_executable(tiscamerasample sample.cpp)
