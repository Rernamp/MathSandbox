target_sources(CommonCode INTERFACE
    main.cpp
)

target_include_directories(CommonCode INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)