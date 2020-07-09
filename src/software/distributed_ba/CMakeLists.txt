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

add_subdirectory(BA)