from vhsdecode.hifi.constants import (
    FORMAT_U8,
    FORMAT_U10_LE,
    FORMAT_U12_LE,
    FORMAT_U16_LE,
    FORMAT_S8,
    FORMAT_S10_LE,
    FORMAT_S12_LE,
    FORMAT_S16_LE,
    FORMAT_F32_LE,
    FORMAT_TO_DTYPE,
    DTYPE_TO_FORMAT
)

# (scale, offset) mapping each raw sample format onto [-1, 1).
_FORMAT_SCALING = {
    FORMAT_U8: (2.0 / 255.0, -1.0),
    FORMAT_U10_LE: (2.0 / 1023.0, -1.0),
    FORMAT_U12_LE: (2.0 / 4095.0, -1.0),
    FORMAT_U16_LE: (2.0 / 65535.0, -1.0),
    FORMAT_S8: (1.0 / 128.0, 0.0),
    FORMAT_S10_LE: (1.0 / 512.0, 0.0),
    FORMAT_S12_LE: (1.0 / 2048.0, 0.0),
    FORMAT_S16_LE: (1.0 / 32768.0, 0.0),
    FORMAT_F32_LE: (1.0, 0.0),
}


def get_normalizer(fmt_or_dtype):
    """Return (normalize, numpy dtype) for a raw format.

    normalize(x, out, n) writes the first n raw samples of x into out as float32; the two
    may share memory since the scaled values are computed before being stored.
    """
    if isinstance(fmt_or_dtype, str):
        string_dtype = fmt_or_dtype.lower()
    else:
        string_dtype = DTYPE_TO_FORMAT[fmt_or_dtype]

    try:
        scale, offset = _FORMAT_SCALING[string_dtype]
    except KeyError:
        raise ValueError(f"Unsupported format: {fmt_or_dtype}")

    def normalize(x, out, n):
        out[:n] = x[:n] * scale + offset

    return normalize, FORMAT_TO_DTYPE[string_dtype]
