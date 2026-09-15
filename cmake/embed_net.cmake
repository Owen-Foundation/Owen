# Embed .o2nn into binary as hex array (like Stockfish)
# Usage: include(embed_net.cmake) then embed_net(nets/o2-final.o2nn)
function(embed_net net_path)
  get_filename_component(net_abs "${net_path}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
  if(NOT EXISTS "${net_abs}")
    message(WARNING "Embed net not found: ${net_abs} — building without embedded net")
    return()
  endif()
  set(gen "${CMAKE_BINARY_DIR}/embedded_net.cpp")
  set(hdr "${CMAKE_BINARY_DIR}/embedded_net.h")
  add_custom_command(
    OUTPUT "${gen}" "${hdr}"
    COMMAND python3 "${CMAKE_SOURCE_DIR}/scripts/embed_net.py" "${net_abs}" "${gen}" "${hdr}"
    DEPENDS "${net_abs}" "${CMAKE_SOURCE_DIR}/scripts/embed_net.py"
    COMMENT "Embedding ${net_path} into binary"
  )
  add_library(embedded_net STATIC "${gen}")
  target_include_directories(embedded_net PUBLIC "${CMAKE_BINARY_DIR}")
  add_dependencies(embedded_net owen2_core)
  target_link_libraries(owen2 PRIVATE -Wl,--whole-archive embedded_net -Wl,--no-whole-archive)
  target_compile_definitions(owen2 PRIVATE OWEN_EMBED_NET)
endfunction()
