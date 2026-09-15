#!/usr/bin/env python3
"""Embed a net file into the binary WITHOUT C++ parsing (OOM-safe for 100MB+ nets).
Emits a tiny TU using the assembler's .incbin (Linux/GAS). Same symbols as before:
  g_embedded_net (bytes), g_embedded_net_size (size_t).
"""
import sys
net, gen, hdr = sys.argv[1:4]
with open(net, 'rb') as f:
    size = len(f.read())
with open(hdr, 'w') as h:
    h.write("#pragma once\n#include <cstddef>\n#include <cstdint>\n")
    h.write("extern const unsigned char g_embedded_net[];\n")
    h.write("extern const unsigned char g_embedded_net_end[];\n")
    h.write(f"// {size} bytes from {net} (via .incbin, Linux/GAS only)\n")
with open(gen, 'w') as g:
    g.write('#include "embedded_net.h"\n')
    g.write('__asm__(\n')
    g.write('".pushsection .rodata\\n"\n')
    g.write('".balign 8\\n"\n')
    g.write('".global g_embedded_net\\n"\n')
    g.write('"g_embedded_net:\\n"\n')
    g.write(f'".incbin \\"{net}\\"\\n"\n')
    g.write('".global g_embedded_net_end\\n"\n')
    g.write('"g_embedded_net_end:\\n"\n')
    g.write('".popsection\\n"\n')
    g.write(');\n')
print(f"Embedded {size} bytes -> {gen} (.incbin)")
