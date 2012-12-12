# The CMakeLists file for Light-SIMD (Testing)

cmake_minimum_required(VERSION 2.8)
project(LightSIMD)

enable_testing()

include_directories(..)
include_directories($ENV{LIGHT_TEST_HOME})

# Compiler configuration

set(CMAKE_BUILD_TYPE "Release")

if (MSVC)
	set(LANG_FLAGS "/arch:SSE2 /DLSSE4_2 /EHsc")
	set(WARNING_FLAGS "/W4")
else (MSVC)
	set(LANG_FLAGS "-std=c++0x -pedantic -march=native ")
	set(WARNING_FLAGS "-Wall -Wextra -Wconversion -Wformat -Wno-unused-parameter ")
endif (MSVC)

if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	set(LANG_FLAGS "${LANG_FLAGS} -stdlib=libc++ -Qunused-arguments")
	set(CMAKE_CXX_COMPILER "clang++")
endif (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")

set(TEST_CXX_FLAGS "${LANG_FLAGS} ${WARNING_FLAGS}")

set(CMAKE_CXX_FLAGS "${TEST_CXX_FLAGS}")

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



# Header file groups (to be used as dependencies)

set(INC ../light_simd)

set(COMMON_BASIC_HS 
    ${INC}/arch.h
    ${INC}/common/common_base.h)
        
set(SSE_BASIC_HS 
    ${INC}/sse/sse_base.h 
    ${INC}/sse/sse_pack.h 
    ${INC}/sse/sse_arith.h
    ${INC}/sse/sse_logical.h
    ${INC}/sse/internal/sse_pack_internal.h)

set(SSE_MATH_HS 
    ${INC}/sse/sse_math.h)

set(SSE_BASIC_DEP_HS
    ${COMMON_BASIC_HS}
    ${SSE_BASIC_HS})
    
set(SSE_MATH_DEP_HS
    ${SSE_BASIC_DEP_HS}
    ${SSE_MATH_HS})
        

# Executables

add_library(test_main STATIC test_main.cpp)

add_executable(test_sse_packs ${SSE_BASIC_DEP_HS} test_sse_packs.cpp)
add_executable(test_sse_arith ${SSE_BASIC_DEP_HS} test_sse_arith.cpp)
add_executable(test_sse_logical ${SSE_BASIC_DEP_HS} test_sse_logical.cpp)

if (SVML_PATH)
add_executable(test_sse_math_svml ${SSE_MATH_DEP_HS} test_sse_math.cpp)
endif (SVML_PATH)

if (LIBM_HOME)
add_executable(test_sse_math_libm ${SSE_MATH_DEP_HS} test_sse_math.cpp)
endif (LIBM_HOME)

target_link_libraries(test_sse_packs test_main)
target_link_libraries(test_sse_arith test_main)
target_link_libraries(test_sse_logical test_main)

if (SVML_PATH)
if (MSVC)
target_link_libraries(test_sse_math_svml test_main ${SVML} ${LIBIRC})
else (MSVC)
target_link_libraries(test_sse_math_svml test_main ${SVML})
endif (MSVC)
endif (SVML_PATH)

if (LIBM_HOME)
target_link_libraries(test_sse_math_libm test_main ${AMDLIBM})
endif (LIBM_HOME)

if (SVML_PATH)
if (LIBM_HOME)
    set(MATH_EXECS test_sse_math_svml test_sse_math_libm)
else (LIBM_HOME)
    set(MATH_EXECS test_sse_math_svml)
endif (LIBM_HOME)
else (SVML_PATH)
if (LIBM_HOME)
    set(MATH_EXECS test_sse_math_libm)
else (LIBM_HOME)
    set(MATH_EXECS )
endif (LIBM_HOME)
endif (SVML_PATH)

set(ALL_EXECUTABLES 
    test_sse_packs
    test_sse_arith
    test_sse_logical
    ${MATH_EXECS})
    
set_target_properties(${ALL_EXECUTABLES}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "bin")    

if (SVML_PATH)
set_target_properties(test_sse_math_svml
	PROPERTIES
	COMPILE_FLAGS "-DLSIMD_USE_INTEL_SVML"
)
endif (SVML_PATH)

if (LIBM_HOME)
set_target_properties(test_sse_math_libm
    PROPERTIES
    COMPILE_FLAGS "-DLSIMD_USE_AMD_LIBM"
)
endif (LIBM_HOME)

# Add Tests

add_test(NAME sse_packs COMMAND test_sse_packs)
add_test(NAME sse_arith COMMAND test_sse_arith)
add_test(NAME sse_logical COMMAND test_sse_logical)

if (SVML_PATH)
add_test(NAME sse_math_svml COMMAND test_sse_math_svml)
endif (SVML_PATH)

if (LIBM_HOME)
add_test(NAME sse_math_libm COMMAND test_sse_math_libm)
endif (LIBM_HOME)


