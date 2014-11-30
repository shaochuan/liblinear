PROJECT(liblinear)
CMAKE_MINIMUM_REQUIRED (VERSION 2.6)
SET(CMAKE_CXX_FLAGS "-g -Wall -Werror -std=c++11 -stdlib=libc++")

OPTION(test "Build all tests." OFF)

ADD_SUBDIRECTORY(proto)

FILE(GLOB BLAS_LIB blas/*.c)

ADD_LIBRARY(blas ${BLAS_LIB})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
ADD_LIBRARY(linear linear.cpp tron.cpp)

TARGET_LINK_LIBRARIES(linear model ${PROTOBUF_LIBRARY})

ADD_EXECUTABLE(train train.cpp)
ADD_EXECUTABLE(predict predict.cpp)

TARGET_LINK_LIBRARIES(train blas linear)
TARGET_LINK_LIBRARIES(predict blas linear)

if (test)
  ADD_SUBDIRECTORY(test)
endif()