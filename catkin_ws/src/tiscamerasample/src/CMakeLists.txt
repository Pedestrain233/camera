cmake_minimum_required(VERSION 2.8.3)
project(tiscamerasample)
add_definitions(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
   roscpp
   rospy
   std_msgs
   message_generation
)

catkin_package(
  LIBRARIES tiscamera
  CATKIN_DEPENDS other_catkin_pkg
  DEPENDS system_lib
  CATKIN_DEPENDS message_runtime
)

include_directories(/home/james/catkin_ws/include)
#include_directories(
#${catkin_INCLUDE_DIRS}
#${ros_INCLUDE_DIRS}
#${glib_INCLUDE_DIRS}
#${gst_INCLUDE_DIRS}
#${opencv_INCLUDE_DIRS}
#${tiscamera_INCLUDE_DIRS}
#)

set(SRC  ${PROJECT_SOURCE_DIR}/tiscamerasample.cpp)

link_directories(${PROJECT_SOURCE_DIR})
add_library(Lib SHARED ${SRC})
target_link_libraries(Lib ${opencv_LIBS})
target_link_libraries(Lib -lgstreamer  -laravis -lstdc++ -lros)

link_directories(${PROJECT_SOURCE_DIR})
add_executable(Test ${SRC})
target_link_libraries(Lib ${opencv_LIBS})
target_link_libraries(Lib -lgstreamer  -laravis -lstdc++ -lros)
