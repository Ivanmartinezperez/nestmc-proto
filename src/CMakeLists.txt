set(HEADERS
    swcio.hpp
)
set(BASE_SOURCES
    common_types_io.cpp
    cell.cpp
    parameter_list.cpp
    profiling/profiler.cpp
    swcio.cpp
    util/debug.cpp
)

if(${WITH_MPI})
    set(BASE_SOURCES ${BASE_SOURCES} communication/mpi.cpp)
endif()

add_library(cellalgo ${BASE_SOURCES} ${HEADERS})
add_dependencies(cellalgo build_all_mods)

