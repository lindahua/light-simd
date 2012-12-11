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

if (SVML_PATH)

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

endif (SVML_PATH)


set(LIBM_HOME "$ENV{AMDLIBM_HOME}")

if ( LIBM_HOME )

    set(AMDLIBM_PATH "$ENV{AMDLIBM_HOME}/lib/dynamic")

    find_library(AMDLIBM amdlibm
        HINTS ${AMDLIBM_PATH})
    
    if (AMDLIBM_NOT_FOUND)
        message(SEND_ERROR "AMD LibM is NOT found.")
    endif (AMDLIBM_NOT_FOUND)

endif ( LIBM_HOME )

# Executables

add_executable(bench_sse_arith bench_sse_arith.cpp)
add_executable(bench_sse_reduce bench_sse_reduce.cpp)

if (SVML_PATH)
add_executable(bench_sse_math_svml bench_sse_math.cpp)
endif (SVML_PATH)

if (LIBM_HOME)
add_executable(bench_sse_math_libm bench_sse_math.cpp)
endif (LIBM_HOME)

if (SVML_PATH)
if (MSVC)
target_link_libraries(bench_sse_math_svml ${SVML} ${LIBIRC})
else (MSVC)
target_link_libraries(bench_sse_math_svml ${SVML})
endif (MSVC)
endif (SVML_PATH)

if (LIBM_HOME)
target_link_libraries(bench_sse_math_libm ${AMDLIBM})
endif (LIBM_HOME)

if (SVML_PATH)
if (LIBM_HOME)
    set(MATH_EXECS bench_sse_math_svml bench_sse_math_libm)
else (LIBM_HOME)
    set(MATH_EXECS bench_sse_math_svml)
endif (LIBM_HOME)
else (SVML_PATH)
if (LIBM_HOME)
    set(MATH_EXECS bench_sse_math_libm)
else (LIBM_HOME)
    set(MATH_EXECS )
endif (LIBM_HOME)
endif (SVML_PATH)

set(ALL_EXECUTABLES
    bench_sse_arith
    bench_sse_reduce
    ${MATH_EXECS})

set_target_properties(${ALL_EXECUTABLES}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "bin")  

if (SVML_PATH)
set_target_properties(bench_sse_math_svml
	PROPERTIES
	COMPILE_FLAGS "-DLSIMD_USE_INTEL_SVML"
)
endif (SVML_PATH)

if (LIBM_HOME)
set_target_properties(bench_sse_math_libm
	PROPERTIES
	COMPILE_FLAGS "-DLSIMD_USE_AMD_LIBM"
)
endif (LIBM_HOME)



