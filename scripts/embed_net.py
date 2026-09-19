#!/usr/bin/env python3
"""Embed a net file into the binary WITHOUT C++ parsing (fast for 100MB+ nets).

Emits a tiny translation unit using the assembler's .incbin directive, so the
168MB net never passes through the C++ parser. Same symbols on all targets:
  g_embedded_net (bytes), g_embedded_net_end (one past the end).

Usage: embed_net.py <net> <gen.cpp> <gen.h> <ELF|MACHO>

Windows/MSVC cannot use inline asm; it embeds via a .rc RCDATA resource
instead (see cmake/embed_net.cmake) and loads it with FindResource.
"""
import sys

net, gen, hdr, target = sys.argv[1:5]
with open(net, 'rb') as f:
    size = len(f.read())

with open(hdr, 'w') as h:
    h.write("#pragma once\n#include <cstddef>\n#include <cstdint>\n")
    h.write("extern const unsigned char g_embedded_net[];\n")
    h.write("extern const unsigned char g_embedded_net_end[];\n")
    h.write(f"// {size} bytes from {net} (via .incbin)\n")

if target == "MACHO":
    # Apple assembler: no .pushsection/.popsection and .globl (not .global);
    # switch the section once (this TU holds nothing else to restore).
    # Mach-O C symbols carry a leading underscore, which hand-written asm
    # must spell out explicitly.
    section_enter = '".section __DATA,__const\\n"'
    section_exit = None
    globl = ".globl"
    sym = "_g_embedded_net"
    sym_end = "_g_embedded_net_end"
else:  # ELF (Linux/GCC/Clang)
    section_enter = '".pushsection .rodata\\n"\n".balign 8\\n"'
    section_exit = '".popsection\\n"'
    globl = ".global"
    sym = "g_embedded_net"
    sym_end = "g_embedded_net_end"

with open(gen, 'w') as g:
    g.write('#include "embedded_net.h"\n')
    g.write('__asm__(\n')
    g.write(f'{section_enter}\n')
    g.write(f'"{globl} {sym}\\n"\n')
    g.write(f'"{sym}:\\n"\n')
    g.write(f'".incbin \\"{net}\\"\\n"\n')
    g.write(f'"{globl} {sym_end}\\n"\n')
    g.write(f'"{sym_end}:\\n"\n')
    if section_exit is not None:
        g.write(f'{section_exit}\n')
    g.write(');\n')
print(f"Embedded {size} bytes -> {gen} (.incbin, {target})")
