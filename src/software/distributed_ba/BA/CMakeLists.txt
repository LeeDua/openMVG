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
        openMVG_geometry
        openMVG_features
        openMVG_graph
        openMVG_matching
        openMVG_multiview
        cereal
        ${OPENMVG_LIBRARY_DEPENDENCIES}
)

target_link_libraries(distributed_ba
    LINK_PRIVATE
        ${CERES_LIBRARIES}
        openMVG_graph
        openMVG_image
        openMVG_lInftyComputerVision
        openMVG_system
        ${STLPLUS_LIBRARY}
)

target_include_directories(distributed_ba
    PRIVATE
    ${CERES_INCLUDE_DIRS}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/openMVG>
)

if (MSVC)
    set_target_properties(distributed_ba PROPERTIES COMPILE_FLAGS "/bigobj")
endif (MSVC)
