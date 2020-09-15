add_executable(graph_export export_visi_graph.cpp)
target_link_libraries(graph_export
        openMVG_system
        openMVG_sfm
        ${STLPLUS_LIBRARY})
target_compile_definitions(graph_export
        PRIVATE -DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")


add_executable(subscene_export subscene_construct.cpp)
target_link_libraries(subscene_export
        openMVG_system
        openMVG_sfm
        ${STLPLUS_LIBRARY})
target_compile_definitions(subscene_export
        PRIVATE -DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
add_subdirectory(BA)

add_executable(main_distributed_BA main_distri_BA.cpp)
target_link_libraries(main_distributed_BA
        PRIVATE
        distributed_ba
        openMVG_system
        openMVG_sfm
        ${STLPLUS_LIBRARY}
        ${MPI_LIBRARIES}
        )


if(MPI_COMPILE_FLAGS)
    set_target_properties(main_distributed_BA PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(main_distributed_BA PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()