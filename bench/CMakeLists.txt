# The CMakeLists file for Light-SIMD (Benchmark)

cmake_minimum_required(VERSION 2.8)
project(LightSIMD_Benchmark)

include_directories(..)

# Compiler configuration

set(CMAKE_BUILD_TYPE "Release")

if (MSVC)
	set(LANG_FLAGS "/arch:SSE2 /EHsc /Ox /fp:fast")
	set(WARNING_FLAGS "/W4")
else (MSVC)
	set(LANG_FLAGS "-pedantic -march=native ")
	set(WARNING_FLAGS "-Wall -Wextra -Wconversion -Wformat -Wno-unused-parameter ")
endif (MSVC)

set(BENCH_CXX_FLAGS "${LANG_FLAGS} ${WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS "${BENCH_CXX_FLAGS}")


# External link libraries

set(SVML_PATH "$ENV{ICC_LIBPATH}")

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	find_library(SVML svml_disp
		HINTS ${SVML_PATH})
	find_library(LIBIRC libirc 
		HINTS ${SVML_PATH})
	
else (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	find_library(SVML svml 
		HINTS ${SVML_PATH})

endif (${CMAKE_SYSTEM_NAME} MATCHES "Windows")

if (SVML_NOT_FOUND)
message(SEND_ERROR "Intel SVML (Short Vector Math Library) is NOT found.")
endif (SVML_NOT_FOUND)


# Executables

add_executable(bench_sse_arith bench_sse_arith.cpp)
add_executable(bench_sse_reduce bench_sse_reduce.cpp)

add_executable(bench_sse_vecs bench_sse_vecs.cpp)
add_executable(bench_sse_mats bench_sse_mats.cpp)
add_executable(bench_sse_mm   bench_sse_mm.cpp)

add_executable(bench_sse_math_svml bench_sse_math.cpp)

if (MSVC)
target_link_libraries(bench_sse_math_svml ${SVML} ${LIBIRC})
else (MSVC)
target_link_libraries(bench_sse_math_svml ${SVML})
endif (MSVC)

set(ALL_EXECUTABLES
    bench_sse_arith
    bench_sse_reduce
    bench_sse_vecs
    bench_sse_mats
    bench_sse_mm
    bench_sse_math_svml)

set_target_properties(${ALL_EXECUTABLES}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "bin")  

set_target_properties(bench_sse_math_svml
	PROPERTIES
	COMPILE_FLAGS "-DLSIMD_USE_INTEL_SVML"
)



