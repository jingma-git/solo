# OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

include_directories(./)