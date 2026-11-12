"""Bit-preserving row copies; no projection, selection or collective changes."""
import triton
import triton.language as tl


@triton.jit
def gather_rows_and_erase(source, selected, packed, size, offset,
                          WIDTH: tl.constexpr, ERASE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < size
    row = tl.load(selected + i // WIDTH, valid, 0)
    position = row * WIDTH + i % WIDTH
    value = tl.load(source + position, valid, 0)
    tl.store(packed + offset + i, value, valid)
    if ERASE:
        tl.store(source + position, 0, valid)


@triton.jit
def scatter_rows(destination, selected, packed, size, offset,
                 WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < size
    row = tl.load(selected + i // WIDTH, valid, 0)
    value = tl.load(packed + offset + i, valid, 0)
    tl.store(destination + row * WIDTH + i % WIDTH, value, valid)
