# Embed .o2nn into the binary so it runs standalone.
# Usage: include(embed_net.cmake) then embed_net(nets/o2-final.o2nn)
#
# Unix (GCC/Clang): a tiny .incbin translation unit (no giant C++ parsing).
# Windows/MSVC: a .rc RCDATA resource (MSVC has no inline asm); main.cpp
#   loads it with FindResource/LoadResource under OWEN_EMBED_WINRC.
function(embed_net net_path)
  get_filename_component(net_abs "${net_path}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
  if(NOT EXISTS "${net_abs}")
    message(WARNING "Embed net not found: ${net_abs} — building without embedded net")
    return()
  endif()

  if(MSVC)
    # NOTE: keep forward slashes — rc.exe processes backslash escapes, so a
    # native D:\... path would corrupt (\n -> newline). Quoted forward-slash
    # paths work fine.
    set(rc "${CMAKE_BINARY_DIR}/baked_net.rc")
    file(WRITE "${rc}" "1 RCDATA \"${net_abs}\"\n")
    target_sources(owen2 PRIVATE "${rc}")
    target_compile_definitions(owen2 PRIVATE OWEN_EMBED_NET OWEN_EMBED_WINRC)
    message(STATUS "Embedding ${net_path} into binary via Win32 resource")
    return()
  endif()

  if(APPLE)
    set(asm_target MACHO)
  else()
    set(asm_target ELF)
  endif()
  set(gen "${CMAKE_BINARY_DIR}/embedded_net.cpp")
  set(hdr "${CMAKE_BINARY_DIR}/embedded_net.h")
  add_custom_command(
    OUTPUT "${gen}" "${hdr}"
    COMMAND python3 "${CMAKE_SOURCE_DIR}/scripts/embed_net.py" "${net_abs}" "${gen}" "${hdr}" "${asm_target}"
    DEPENDS "${net_abs}" "${CMAKE_SOURCE_DIR}/scripts/embed_net.py"
    COMMENT "Embedding ${net_path} into binary"
  )
  add_library(embedded_net STATIC "${gen}")
  target_include_directories(embedded_net PUBLIC "${CMAKE_BINARY_DIR}")
  # main.cpp references g_embedded_net, so plain static linking pulls the TU in.
  target_link_libraries(owen2 PRIVATE embedded_net)
  target_compile_definitions(owen2 PRIVATE OWEN_EMBED_NET)
endfunction()
