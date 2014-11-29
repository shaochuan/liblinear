PROJECT(liblinear)
CMAKE_MINIMUM_REQUIRED (VERSION 2.6)
SET(CMAKE_CXX_FLAGS "-g -Wall -Werror -std=c++11 -stdlib=libc++")

option(test "Build all tests." OFF)

FILE(GLOB BLAS_LIB blas/*.c)

ADD_LIBRARY(blas ${BLAS_LIB})
ADD_LIBRARY(linear linear.cpp tron.cpp)

ADD_EXECUTABLE(train train.cpp)
ADD_EXECUTABLE(predict predict.cpp)

TARGET_LINK_LIBRARIES(train blas linear)
TARGET_LINK_LIBRARIES(predict blas linear)

if (test)
  ADD_SUBDIRECTORY(test)
endif()