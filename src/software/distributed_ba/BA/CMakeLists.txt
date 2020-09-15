file(
    GLOB_RECURSE
    distributed_ba_headers
    *.hpp
)
file(
    GLOB_RECURSE
    distributed_ba_cpp
    *.cpp
)


add_library(distributed_ba ${distributed_ba_headers} ${distributed_ba_cpp})
target_compile_features(distributed_ba INTERFACE ${CXX11_FEATURES})

target_link_libraries(distributed_ba
    PUBLIC
        openMVG_sfm
        openMVG_system
        cereal
        ${OPENMVG_LIBRARY_DEPENDENCIES}
    PRIVATE
        ${MPI_LIBRARIES}
)

target_link_libraries(distributed_ba
    LINK_PUBLIC
        ${CERES_LIBRARIES}
        ${STLPLUS_LIBRARY}

)

if(MPI_COMPILE_FLAGS)
    set_target_properties(distributed_ba PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(distributed_ba PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

target_include_directories(distributed_ba
    PUBLIC
    ${CERES_INCLUDE_DIRS}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/openMVG>
)

if (MSVC)
    set_target_properties(distributed_ba PROPERTIES COMPILE_FLAGS "/bigobj")
endif (MSVC)
