from multiprocessing.shared_memory import SharedMemory
import numba
import numpy as np
from dataclasses import dataclass

import string
from random import SystemRandom

import ctypes
import mmap
import os
from functools import lru_cache
from itertools import cycle
import atexit


REAL_DTYPE = np.float32
ALIGNMENT = 64

NumbaAudioArray = numba.types.Array(numba.types.float32, 1, "C")

class NUMA:
    # Memory binding is performed via libnuma's numa_tonode_memory wrapper,
    # which issues the correct mbind syscall per architecture. The previous
    # raw libc.syscall(SYS_mbind=237, ...) was x86_64-only and would invoke
    # the wrong syscall number on other architectures (e.g. aarch64).

    _libnuma = None
    _libc = None

    class Bitmask(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_ulong),
            ("maskp", ctypes.POINTER(ctypes.c_ulong)),
        ]

    @classmethod
    def _load_libnuma(cls):
        if cls._libnuma is not None:
            return cls._libnuma

        try:
            libnuma = ctypes.CDLL("libnuma.so.1")

            libnuma.numa_available.restype = ctypes.c_int
            libnuma.numa_max_node.restype = ctypes.c_int
            libnuma.numa_node_to_cpus.argtypes = [ctypes.c_int, ctypes.c_void_p]
            libnuma.numa_node_to_cpus.restype = ctypes.c_int
            libnuma.numa_run_on_node.argtypes = [ctypes.c_int]
            libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
            libnuma.numa_tonode_memory.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            libnuma.numa_tonode_memory.restype = ctypes.c_int

            # Bitmask API used by bind_process: these return / take
            # `struct bitmask *` pointers. Without an explicit restype the
            # default c_int truncates the 64-bit pointer returned by
            # numa_allocate_nodemask to 32 bits, yielding an invalid pointer
            # that segfaults when the bitmask helpers dereference it.
            # (Linux-only: on Windows libnuma is absent so this path is
            # never reached -- which is why the crash only shows on Linux.)
            libnuma.numa_allocate_nodemask.restype = ctypes.c_void_p
            libnuma.numa_bitmask_clearall.argtypes = [ctypes.c_void_p]
            libnuma.numa_bitmask_clearall.restype = ctypes.c_void_p
            libnuma.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            libnuma.numa_bitmask_setbit.restype = ctypes.c_void_p
            libnuma.numa_run_on_node_mask.argtypes = [ctypes.c_void_p]
            libnuma.numa_run_on_node_mask.restype = ctypes.c_int
            libnuma.numa_bitmask_free.argtypes = [ctypes.c_void_p]

            cls._libnuma = libnuma
            return libnuma

        except Exception:
            cls._libnuma = None
            return None

    @classmethod
    def _load_libc(cls):
        if cls._libc is not None:
            return cls._libc

        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)

            cls._libc = libc
            return libc

        except Exception:
            cls._libc = None
            return None

    @classmethod
    def _available(cls):
        libnuma = cls._load_libnuma()
        libc = cls._load_libc()
        return bool(libc and libnuma and libnuma.numa_available() >= 0)

    @classmethod
    @lru_cache(maxsize=1)
    def _nodes(cls):
        if not cls._available():
            return cycle([0])

        try:
            max_node = cls._libnuma.numa_max_node()
            nodes = [n for n in range(max_node + 1) if cls._cpus_for_node(n)]
            return cycle(nodes or [0])
        except Exception:
            return cycle([0])

    @classmethod
    @lru_cache(maxsize=128)
    def _cpus_for_node(cls, node: int):
        if not cls._available():
            return None

        lib = cls._libnuma
        MAX_CPUS = 1024

        words = MAX_CPUS // (8 * ctypes.sizeof(ctypes.c_ulong))
        mask = (ctypes.c_ulong * words)()

        bm = cls.Bitmask()
        bm.size = MAX_CPUS
        bm.maskp = mask

        if lib.numa_node_to_cpus(node, ctypes.byref(bm)) != 0:
            return None

        cpus = set()
        bits = 8 * ctypes.sizeof(ctypes.c_ulong)

        for cpu in range(MAX_CPUS):
            word = mask[cpu // bits]
            bit = cpu % bits
            if word & (1 << bit):
                cpus.add(cpu)

        return cpus

    @classmethod
    def next_node(cls):
        return next(cls._nodes())

    @classmethod
    def bind_process(cls, node: int) -> bool:
        if not cls._available():
            return False

        cpus = cls._cpus_for_node(node)
        if not cpus:
            return False

        try:
            cpus = cls._cpus_for_node(node)
            if cpus:
                os.sched_setaffinity(0, cpus)

            mask = cls._libnuma.numa_allocate_nodemask()
            try:
                cls._libnuma.numa_bitmask_clearall(mask)
                cls._libnuma.numa_bitmask_setbit(mask, node)

                cls._libnuma.numa_run_on_node_mask(mask)
                cls._libnuma.numa_set_preferred(node)
                return True
            finally:
                # avoid leaking a nodemask on every bind_process call
                cls._libnuma.numa_bitmask_free(mask)
        except Exception:
            return False

    @classmethod
    def get_shared_memory(
        cls,
        name: str,
        size: int,
        numa_node: int | None = None,
    ):
        shm = SharedMemory(name=name, size=size, create=True)

        if cls._available() and numa_node is not None:
            try:
                buf = shm.buf

                addr = ctypes.addressof(
                    ctypes.c_char.from_buffer(buf)
                )

                # Bind this shared-memory range to the NUMA node using
                # libnuma's numa_tonode_memory wrapper. This is
                # architecture-portable (libnuma issues the correct mbind
                # syscall per-arch) -- the previous raw
                # libc.syscall(SYS_mbind=237, ...) was x86_64-only and
                # would invoke the wrong syscall number on aarch64/other.
                cls._libnuma.numa_tonode_memory(
                    ctypes.c_void_p(addr),
                    ctypes.c_size_t(len(buf)),
                    ctypes.c_int(numa_node),
                )

                # fault the pages in under the new policy so they are
                # physically allocated on the bound node
                for offset in range(0, len(buf), mmap.PAGESIZE):
                    buf[offset] = 0

            except Exception:
                pass

        return shm

@dataclass
class DecoderState:
    def __init__(self, decoder, buffer_name, block_frames_read, block_size, block_num, is_last_block, numa_node):
        block_sizes = decoder.set_block_sizes(block_size)
        block_overlap = decoder.get_block_overlap()

        self.name = buffer_name
        self.numa_node = numa_node
        self.block_num = block_num
        self.is_last_block = is_last_block

        # block data for input rf
        self.block_frames_read = block_frames_read
        self.block_dtype = REAL_DTYPE
        self.block_size = block_sizes["block_size"]
        self.block_overlap = block_overlap["block_overlap"]
        self.block_read_overlap = block_overlap["block_read_overlap"]

        # block data for demodulated audio @ 192000Hz
        self.audio_dtype = REAL_DTYPE
        self.block_audio_size = block_sizes["block_audio_size"]

        # block data for resampled audio @ user set audio rate
        self.block_audio_final_size = block_sizes["block_audio_final_size"]
        self.block_audio_final_overlap = block_overlap["block_audio_final_overlap"]

    name: str
    numa_node: int
    block_num: int
    is_last_block: bool

    block_frames_read: int
    block_dtype: np.dtype
    block_size: int
    block_overlap: int
    block_read_overlap: int

    audio_dtype: np.dtype
    block_audio_size: int

    block_audio_final_size: int
    block_audio_final_overlap: int
        
    @property
    def block_audio_final_len(self):
        if self.is_last_block:
            # shrink the final stereo output to only include the actual frames read, and the overlap that would have been used for the next block
            rf_rate_to_final_rate_ratio = self.block_size / self.block_audio_final_size
            audio_size = round(self.block_frames_read / rf_rate_to_final_rate_ratio)
            return max(50, audio_size + self.block_audio_final_overlap)
        else:
            # don't allow 0 or negative length audio, even if there's more overlap than actual audio
            return max(50, self.block_audio_final_size - self.block_audio_final_overlap * 2)


def to_aligned_offset(size):
    alignment = ALIGNMENT
    offset = size % alignment
    aligned_size = 0 if offset == 0 else alignment - offset
    return size + aligned_size


class PostProcessorSharedMemory:
    def __init__(self, decoder_state: DecoderState):
        self.shared_memory = SharedMemory(name=decoder_state.name)

        self.size = self.shared_memory.size
        self.buf = self.shared_memory.buf
        self.name = self.shared_memory.name
        self.close = self.shared_memory.close
        self.unlink = self.shared_memory.unlink

        self.numa_node = decoder_state.numa_node
        self.audio_dtype = decoder_state.audio_dtype
        self.channel_len = decoder_state.block_audio_final_len
        self.audio_dtype_item_size = np.dtype(self.audio_dtype).itemsize

        ### Post Processing Memory
        # |--pre_left--|--pre_right--|--post_left--|--post_right--|
        # |-----------stereo---------|

        # pre left
        self.l_pre_offset = 0
        self.l_pre_len = self.channel_len
        self.l_pre_bytes = self.l_pre_len * self.audio_dtype_item_size
        # pre right
        self.r_pre_offset = to_aligned_offset(self.l_pre_offset + self.l_pre_bytes)
        self.r_pre_len = self.channel_len
        self.r_pre_bytes = self.r_pre_len * self.audio_dtype_item_size

        # overlaps with pre
        ## stereo out
        self.stereo_offset = 0
        self.stereo_len = self.channel_len * 2
        self.stereo_bytes = self.stereo_len * self.audio_dtype_item_size

        ## noise reduction out
        # left
        self.l_post_offset = to_aligned_offset(
            max(
                self.stereo_offset + self.stereo_bytes,
                self.r_pre_offset + self.r_pre_bytes,
            )
        )
        self.l_post_len = self.channel_len
        self.l_post_bytes = self.l_post_len * self.audio_dtype_item_size
        # right
        self.r_post_offset = to_aligned_offset(self.l_post_offset + self.l_post_bytes)
        self.r_post_len = self.channel_len
        self.r_post_bytes = self.r_post_len * self.audio_dtype_item_size

    @staticmethod
    def get_shared_memory(channel_size, name, audio_dtype=REAL_DTYPE, numa_node=None):
        byte_size = (
            to_aligned_offset(channel_size * np.dtype(audio_dtype).itemsize * 4)
            + ALIGNMENT * 16
        )

        # allow more than one instance to run at a time
        system_random = SystemRandom()
        name += "_" + "".join(
            system_random.choice(string.ascii_lowercase + string.digits)
            for _ in range(8)
        )

        # this instance must be saved in a variable that persists on both processes
        # Windows will remove the shared memory if it garbage collects the handle in any of the processes it is open in
        # https://stackoverflow.com/a/63717188
        return NUMA.get_shared_memory(name, byte_size, numa_node)

    def get_pre_left(self) -> np.array:
        return np.ndarray(
            self.l_pre_len,
            dtype=self.audio_dtype,
            offset=self.l_pre_offset,
            buffer=self.buf,
            order="C"
        )

    def get_pre_right(self) -> np.array:
        return np.ndarray(
            self.r_pre_len,
            dtype=self.audio_dtype,
            offset=self.r_pre_offset,
            buffer=self.buf,
            order="C"
        )

    # overlaps with the pre audio
    def get_stereo(self) -> np.array:
        return np.ndarray(
            self.stereo_len,
            dtype=self.audio_dtype,
            offset=self.stereo_offset,
            buffer=self.buf,
            order="C"
        )

    def get_post_left(self) -> np.array:
        return np.ndarray(
            self.l_post_len,
            dtype=self.audio_dtype,
            offset=self.l_post_offset,
            buffer=self.buf,
            order="C"
        )

    def get_post_right(self) -> np.array:
        return np.ndarray(
            self.r_post_len,
            dtype=self.audio_dtype,
            offset=self.r_post_offset,
            buffer=self.buf,
            order="C"
        )


class DecoderSharedMemory:
    def __init__(self, decoder_state: DecoderState):
        self.shared_memory = SharedMemory(name=decoder_state.name)

        self.size = self.shared_memory.size
        self.buf = self.shared_memory.buf
        self.name = self.shared_memory.name
        self.close = self.shared_memory.close
        self.unlink = self.shared_memory.unlink

        self.numa_node = decoder_state.numa_node
        self.block_dtype = decoder_state.block_dtype
        self.block_dtype_item_size = np.dtype(self.block_dtype).itemsize

        self.audio_dtype = decoder_state.audio_dtype
        self.audio_dtype_item_size = np.dtype(self.audio_dtype).itemsize

        self.block_audio_final_len = decoder_state.block_audio_final_len

        ### Decoder Memory
        # -------------------------------raw_data-------------------------------|
        # RF data is demodulated and raw data can be discarded
        # Output audio data overwrites where the raw data was
        # |--pre_left--|--pre_right--|-------------------empty------------------|
        # |--pre_left--|--pre_right--|-------------------empty------------------|

        ## raw data in
        # first overlap
        self.block_start_overlap_offset = 0
        self.block_start_overlap_len = decoder_state.block_read_overlap
        self.block_start_overlap_bytes = (
            self.block_start_overlap_len * self.block_dtype_item_size
        )
        # block data
        self.block_frames_read = decoder_state.block_frames_read
        self.block_offset = (
            self.block_start_overlap_offset + self.block_start_overlap_bytes
        )
        self.block_len = decoder_state.block_size - (
            decoder_state.block_read_overlap * 2
        )
        self.block_bytes = self.block_len * self.block_dtype_item_size
        # second overlap
        self.block_end_overlap_offset = self.block_offset + self.block_bytes
        self.block_end_overlap_len = decoder_state.block_read_overlap
        self.block_end_overlap_bytes = (
            self.block_end_overlap_len * self.block_dtype_item_size
        )

        # pre left
        self.l_pre_offset = 0
        self.l_pre_len = self.block_audio_final_len
        self.l_pre_bytes = self.l_pre_len * self.audio_dtype_item_size
        # pre right
        self.r_pre_offset = to_aligned_offset(self.l_pre_offset + self.l_pre_bytes)
        self.r_pre_len = self.block_audio_final_len
        self.r_pre_bytes = self.r_pre_len * self.audio_dtype_item_size

    @staticmethod
    def get_shared_memory(
        block_size,
        block_overlap,
        block_audio_final_size,
        name,
        numa_node=None,
    ):
        max_audio_size = (
            block_audio_final_size + to_aligned_offset(block_audio_final_size)
        ) * np.dtype(REAL_DTYPE).itemsize
        block_size = (block_size + block_overlap * 2) * np.dtype(REAL_DTYPE).itemsize

        byte_size = max(max_audio_size, block_size)

        # allow more than one instance to run at a time
        system_random = SystemRandom()
        name += "_" + "".join(
            system_random.choice(string.ascii_lowercase + string.digits)
            for _ in range(8)
        )

        # this instance must be saved in a variable that persists on both processes
        # Windows will remove the shared memory if it garbage collects the handle in any of the processes it is open in
        # https://stackoverflow.com/a/63717188
        return NUMA.get_shared_memory(name, byte_size, numa_node)

    ## Decoder methods

    # block data with start and end overlap included
    def get_block(self) -> np.array:
        return np.ndarray(
            self.block_start_overlap_len + self.block_len + self.block_end_overlap_len,
            dtype=self.block_dtype,
            offset=self.block_start_overlap_offset,
            buffer=self.buf,
            order="C"
        )

    # block starts after first overlap, goes until the end of the last overlap
    # first part of the block is copied from the previous read
    def get_block_in(self) -> np.array:
        return np.ndarray(
            self.block_len + self.block_end_overlap_len,
            dtype=self.block_dtype,
            offset=self.block_offset,
            buffer=self.buf,
            order="C"
        )

    # end overlap is copied into the start overlap
    def get_block_in_start_overlap(self) -> np.array:
        return np.ndarray(
            self.block_start_overlap_len,
            dtype=self.block_dtype,
            offset=self.block_start_overlap_offset,
            buffer=self.buf,
            order="C"
        )

    # end overlap is copied and appended to the beginning of the next block
    def get_block_in_end_overlap(self) -> np.array:
        return np.ndarray(
            self.block_end_overlap_len,
            dtype=self.block_dtype,
            offset=self.block_end_overlap_offset,
            buffer=self.buf,
            order="C"
        )

    def get_pre_left(self) -> np.array:
        return np.ndarray(
            self.block_audio_final_len,
            dtype=self.audio_dtype,
            offset=self.l_pre_offset,
            buffer=self.buf,
            order="C"
        )

    def get_pre_right(self) -> np.array:
        return np.ndarray(
            self.block_audio_final_len,
            dtype=self.audio_dtype,
            offset=self.r_pre_offset,
            buffer=self.buf,
            order="C"
        )

def copy_data(src, dst, length, src_offset=0, dst_offset=0):
    """Copy length samples, or as many as both buffers hold."""
    n = min(length, len(src) - src_offset, len(dst) - dst_offset)
    dst[dst_offset : dst_offset + n] = src[src_offset : src_offset + n]


class PeakGain(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_float),
        ("right", ctypes.c_float),
    ]

def cleanup_process(process):
    atexit.unregister(process.terminate)
    atexit.unregister(process.join)
    process.terminate()
    process.join()
