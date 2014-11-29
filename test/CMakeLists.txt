CMAKE_MINIMUM_REQUIRED (VERSION 2.6)
PROJECT(liblinear_test)
SET(CMAKE_CXX_FLAGS "-g -Wall -Werror -std=c++11 -stdlib=libc++")

ADD_SUBDIRECTORY (gtest-1.7.0)

ENABLE_TESTING()
INCLUDE_DIRECTORIES(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

FILE(GLOB TEST_SRCS *.cpp)
ADD_EXECUTABLE(liblinear_test ${TEST_SRCS})
TARGET_LINK_LIBRARIES(liblinear_test gtest gtest_main)

ADD_TEST(AllLibnearTests liblinear_test)