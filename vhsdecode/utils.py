import numpy as np


def gen_wave_at_frequency(frequency, sample_frequency, num_samples, gen_func=np.sin, dtype=np.float32):
    """Generate a sine wave with the specified parameters."""
    samples = np.arange(num_samples)
    wave_scale = frequency / sample_frequency
    return gen_func(2 * np.pi * wave_scale * samples, dtype=dtype)
