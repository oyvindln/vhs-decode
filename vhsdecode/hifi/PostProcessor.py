import atexit
from multiprocessing import (
    Pipe,
    Process,
    current_process,
)

from vhsdecode.hifi.utils import (
    DecoderSharedMemory,
    PostProcessorSharedMemory,
    NumbaAudioArray,
    REAL_DTYPE,
    cleanup_process
)

from numba import njit
import numba
import atexit
from setproctitle import setproctitle

import numpy as np

from vhsdecode.hifi.utils import (
    DecoderSharedMemory,
    PostProcessorSharedMemory,
    NumbaAudioArray,
    REAL_DTYPE,
    cleanup_process
)
from vhsdecode.hifi.HiFiDecode import (
    DCBlocker,
    Deemphasis,
    Expander,
    SpectralNoiseReduction,
)


class PostProcessor:
    def __init__(
        self,
        decode_options: dict,
        decoder_out_queue,
        channel_size,
        post_processor_shared_memory_idle_queue,
        decoder_shared_memory_idle_queue,
        blocks_enqueued,
        out_conn,
        peak_gain,
        numa_node
    ):
        self.numa_node = numa_node
        self.final_audio_rate = decode_options["audio_rate"]
        self.enable_expander = decode_options["enable_expander"]
        self.enable_deemphasis = decode_options["enable_deemphasis"]
        self.spectral_nr_amount = decode_options["spectral_nr_amount"]
        self.format = decode_options["format"]
        self.peak_gain = peak_gain

        # create processes and wire up queues
        #
        #                                 (left channel)
        #                               spectral_noise_reduction_worker --> expander_worker
        #                             /                                                            \
        # data in --[block_sorter]----                                                              --> mix_to_stereo_worker --> data out
        #                             \   (right channel)                                          /
        #                               spectral_noise_reduction_worker --> expander_worker

        self.decoder_out_queue = decoder_out_queue
        self.mix_to_stereo_worker_output = out_conn
        self.blocks_enqueued = blocks_enqueued
        self.decoder_shared_memory_idle_queue = decoder_shared_memory_idle_queue

        self.post_processor_shared_memory = []
        self.post_processor_shared_memory_idle_queue = (
            post_processor_shared_memory_idle_queue
        )
        self.post_processor_num_shared_memory = 16
        for i in range(self.post_processor_num_shared_memory):
            shared_memory = PostProcessorSharedMemory.get_shared_memory(
                channel_size, f"hifi_post_mem_{i}", numa_node=self.numa_node
            )
            self.post_processor_shared_memory.append(shared_memory)
            self.post_processor_shared_memory_idle_queue.put((shared_memory.name, self.numa_node))
            atexit.register(shared_memory.close)
            atexit.register(shared_memory.unlink)

        block_sort_l_in_rx, block_sort_l_in_tx = Pipe(duplex=False)
        block_sort_r_in_rx, block_sort_r_in_tx = Pipe(duplex=False)
        self.block_sorter_process = Process(
            target=PostProcessor.block_sorter_worker,
            name="hifi_block_sort",
            args=(
                self.decoder_out_queue,
                self.decoder_shared_memory_idle_queue,
                self.blocks_enqueued,
                self.post_processor_shared_memory_idle_queue,
                block_sort_l_in_tx,
                block_sort_r_in_tx,
            ),
        )
        self.block_sorter_process.start()
        atexit.register(self.block_sorter_process.terminate)
        atexit.register(self.block_sorter_process.join)

        dc_blocker_worker_l_rx, dc_blocker_worker_l_tx = Pipe(duplex=False)
        self.dc_blocker_worker_l = Process(
            target=PostProcessor.dc_block_worker,
            name="hifi_dc_block_l",
            args=(
                block_sort_l_in_rx,
                dc_blocker_worker_l_tx,
                self.final_audio_rate,
            ),
        )
        self.dc_blocker_worker_l.start()
        atexit.register(self.dc_blocker_worker_l.terminate)
        atexit.register(self.dc_blocker_worker_l.join)

        dc_blocker_worker_r_rx, dc_blocker_worker_r_tx = Pipe(duplex=False)
        self.dc_blocker_worker_r = Process(
            target=PostProcessor.dc_block_worker,
            name="hifi_dc_block_r",
            args=(
                block_sort_r_in_rx,
                dc_blocker_worker_r_tx,
                self.final_audio_rate,
            ),
        )
        self.dc_blocker_worker_r.start()
        atexit.register(self.dc_blocker_worker_r.terminate)
        atexit.register(self.dc_blocker_worker_r.join)

        spectral_nr_worker_l_rx, spectral_nr_worker_l_tx = Pipe(duplex=False)
        self.spectral_nr_worker_l = Process(
            target=PostProcessor.spectral_noise_reduction_worker,
            name="hifi_spec_nr_l",
            args=(
                dc_blocker_worker_l_rx,
                spectral_nr_worker_l_tx,
                self.spectral_nr_amount,
                self.final_audio_rate,
            ),
        )
        self.spectral_nr_worker_l.start()
        atexit.register(self.spectral_nr_worker_l.terminate)
        atexit.register(self.spectral_nr_worker_l.join)

        spectral_nr_worker_r_rx, spectral_nr_worker_r_tx = Pipe(duplex=False)
        self.spectral_nr_worker_r = Process(
            target=PostProcessor.spectral_noise_reduction_worker,
            name="hifi_spec_nr_r",
            args=(
                dc_blocker_worker_r_rx,
                spectral_nr_worker_r_tx,
                self.spectral_nr_amount,
                self.final_audio_rate,
            ),
        )
        self.spectral_nr_worker_r.start()
        atexit.register(self.spectral_nr_worker_r.terminate)
        atexit.register(self.spectral_nr_worker_r.join)

        expander_worker = PostProcessor.expander_8mm_worker if self.format == "8mm" else PostProcessor.expander_vhs_worker

        expander_worker_l_out_rx, expander_worker_l_out_tx = Pipe(duplex=False)
        self.expander_worker_l = Process(
            target=expander_worker,
            name="hifi_expander_l",
            args=(
                spectral_nr_worker_l_rx,
                expander_worker_l_out_tx,
                self.enable_deemphasis,
                self.enable_expander,
                self.final_audio_rate,
                decode_options["deemphasis_low_tau"],
                decode_options["deemphasis_high_tau"],
                decode_options["nr_deemphasis_low_tau"],
                decode_options["nr_deemphasis_high_tau"],
                decode_options["expander_gain"],
                decode_options["expander_ratio"],
                decode_options["expander_env_detection"],
                decode_options["expander_attack_tau"],
                decode_options["expander_hold_tau"],
                decode_options["expander_release_tau"],
                decode_options["expander_weighting_low_tau"],
                decode_options["expander_weighting_high_tau"],
                decode_options["expander_weighting_low_pass"],
                decode_options["expander_weighting_low_pass_transition"]
            ),
        )
        self.expander_worker_l.start()
        atexit.register(self.expander_worker_l.terminate)
        atexit.register(self.expander_worker_l.join)

        expander_worker_r_out_rx, expander_worker_r_out_tx = Pipe(duplex=False)
        self.expander_worker_r = Process(
            target=expander_worker,
            name="hifi_expander_r",
            args=(
                spectral_nr_worker_r_rx,
                expander_worker_r_out_tx,
                self.enable_deemphasis,
                self.enable_expander,
                self.final_audio_rate,
                decode_options["deemphasis_low_tau"],
                decode_options["deemphasis_high_tau"],
                decode_options["nr_deemphasis_low_tau"],
                decode_options["nr_deemphasis_high_tau"],
                decode_options["expander_gain"],
                decode_options["expander_ratio"],
                decode_options["expander_env_detection"],
                decode_options["expander_attack_tau"],
                decode_options["expander_hold_tau"],
                decode_options["expander_release_tau"],
                decode_options["expander_weighting_low_tau"],
                decode_options["expander_weighting_high_tau"],
                decode_options["expander_weighting_low_pass"],
                decode_options["expander_weighting_low_pass_transition"]
            ),
        )
        self.expander_worker_r.start()
        atexit.register(self.expander_worker_r.terminate)
        atexit.register(self.expander_worker_r.join)

        self.mix_to_stereo_worker_process = Process(
            target=PostProcessor.mix_to_stereo_worker,
            name="hifi_stereo_mix",
            args=(
                expander_worker_l_out_rx,
                expander_worker_r_out_rx,
                self.mix_to_stereo_worker_output,
                self.peak_gain,
                self.final_audio_rate,
            ),
        )
        self.mix_to_stereo_worker_process.start()
        atexit.register(self.mix_to_stereo_worker_process.terminate)
        atexit.register(self.mix_to_stereo_worker_process.join)

    @staticmethod
    def dc_block_worker(
        in_conn,
        out_conn,
        final_audio_rate
    ):
        setproctitle(current_process().name)
        dc_blocker = DCBlocker(
            final_audio_rate,
            1
        )

        while True:
            while True:
                try:
                    decoder_state, channel_num = in_conn.recv()
                    break
                except InterruptedError:
                    pass
                except EOFError:
                    return

            buffer = PostProcessorSharedMemory(decoder_state)
            if channel_num == 0:
                pre = buffer.get_pre_left()
            else:
                pre = buffer.get_pre_right()

            if decoder_state.block_num == 0:
                # prime the state
                dc_blocker.process(pre.copy())

            dc_blocker.process(pre)

            buffer.close()
            out_conn.send((decoder_state, channel_num))
        
    @staticmethod
    def spectral_noise_reduction_worker(
        in_conn,
        out_conn,
        spectral_nr_amount,
        final_audio_rate,
    ):
        setproctitle(current_process().name)

        if spectral_nr_amount > 0:
            spectral_nr = SpectralNoiseReduction(
                nr_reduction_amount=spectral_nr_amount,
                audio_rate=final_audio_rate,
            )

        while True:
            while True:
                try:
                    decoder_state, channel_num = in_conn.recv()
                    break
                except InterruptedError:
                    pass
                except EOFError:
                    return

            buffer = PostProcessorSharedMemory(decoder_state)
            if channel_num == 0:
                pre = buffer.get_pre_left()
                spectral_nr_out = buffer.get_post_left()
            else:
                pre = buffer.get_pre_right()
                spectral_nr_out = buffer.get_post_right()

            if spectral_nr_amount > 0:
                spectral_nr.spectral_nr(pre, spectral_nr_out)
            else:
                spectral_nr_out[:] = pre[: len(spectral_nr_out)]

            buffer.close()
            out_conn.send((decoder_state, channel_num))

    @staticmethod
    def expander_vhs_worker(
        in_conn,
        out_conn,
        enable_deemphasis,
        enable_expander,
        final_audio_rate,
        deemphasis_low_tau,
        deemphasis_high_tau,
        nr_deemphasis_low_tau,
        nr_deemphasis_high_tau,
        expander_gain,
        expander_ratio,
        expander_env_detection,
        expander_attack_tau,
        expander_hold_tau,
        expander_release_tau,
        expander_weighting_low_tau,
        expander_weighting_high_tau,
        expander_weighting_low_pass,
        expander_weighting_low_pass_transition,
    ):
        setproctitle(current_process().name)
        deemphasis_pre_1 = Deemphasis(
            final_audio_rate,
            deemphasis_low_tau,
            deemphasis_high_tau,
        )
        deemphasis_pre_2 = Deemphasis(
            final_audio_rate,
            deemphasis_low_tau,
            deemphasis_high_tau,
        )
        nr_deemphasis = Deemphasis(
            final_audio_rate,
            nr_deemphasis_low_tau,
            nr_deemphasis_high_tau,
        )
        expander = Expander(
            final_audio_rate,
            expander_gain,
            expander_ratio,
            expander_env_detection,
            expander_attack_tau,
            expander_hold_tau,
            expander_release_tau,
            expander_weighting_low_tau,
            expander_weighting_high_tau,
            expander_weighting_low_pass,
            expander_weighting_low_pass_transition,
        )

        while True:
            while True:
                try:
                    decoder_state, channel_num = in_conn.recv()
                    break
                except InterruptedError:
                    pass
                except EOFError:
                    return

            buffer = PostProcessorSharedMemory(decoder_state)
            if channel_num == 0:
                pre = buffer.get_pre_left()
                post = buffer.get_post_left()
            else:
                pre = buffer.get_pre_right()
                post = buffer.get_post_right()

            if enable_deemphasis:
                # first deemphasis stage happens before the noise reduction block
                # IEC 60774-2 Figure 2, pg.15 (pre-emphasis parameters)
                # IEC 60774-2 Figure 4, pg.17 (pre-emphasis location)
                deemphasis_pre_1.process(pre)
                deemphasis_pre_2.process(post)

                # second deemphasis stage only happens on the audio (not the weighted input)
                # IEC 60774-2 Figure 5, pg.19 (noise reduction layout)
                nr_deemphasis.process(post)

            if enable_expander:
                if decoder_state.block_num == 0:
                    # prime the expander's gain if this is the first block
                    expander.process(np.copy(pre, order="C"), np.copy(post, order="C"))
                expander.process(pre, post)

            buffer.close()
            out_conn.send(decoder_state)

    @staticmethod
    def expander_8mm_worker(
        in_conn,
        out_conn,
        enable_deemphasis,
        enable_expander,
        final_audio_rate,
        deemphasis_low_tau,
        deemphasis_high_tau,
        nr_deemphasis_low_tau,
        nr_deemphasis_high_tau,
        expander_gain,
        expander_ratio,
        expander_env_detection,
        expander_attack_tau,
        expander_hold_tau,
        expander_release_tau,
        expander_weighting_low_tau,
        expander_weighting_high_tau,
        expander_weighting_low_pass,
        expander_weighting_low_pass_transition,
    ):
        setproctitle(current_process().name)
        deemphasis_2 = Deemphasis(
            final_audio_rate,
            deemphasis_low_tau,
            deemphasis_high_tau,
        )
        deemphasis_1 = Deemphasis(
            final_audio_rate,
            nr_deemphasis_low_tau,
            nr_deemphasis_high_tau,
        )
        expander = Expander(
            final_audio_rate,
            expander_gain,
            expander_ratio,
            expander_env_detection,
            expander_attack_tau,
            expander_hold_tau,
            expander_release_tau,
            expander_weighting_low_tau,
            expander_weighting_high_tau,
            expander_weighting_low_pass,
            expander_weighting_low_pass_transition,
        )

        while True:
            while True:
                try:
                    decoder_state, channel_num = in_conn.recv()
                    break
                except InterruptedError:
                    pass
                except EOFError:
                    return

            buffer = PostProcessorSharedMemory(decoder_state)
            if channel_num == 0:
                pre = buffer.get_pre_left()
                post = buffer.get_post_left()
            else:
                pre = buffer.get_pre_right()
                post = buffer.get_post_right()

            # IEC 60843-1-1993 Figure 34, pg.101
            if enable_deemphasis:
                deemphasis_2.process(post)

            if enable_expander:
                if decoder_state.block_num == 0:
                    # prime the expander's gain if this is the first block
                    expander.process(pre, np.copy(post, order="C"))
                expander.process(pre, post)

            if enable_deemphasis:
                # reverse noise reduction pre-emphasis
                deemphasis_1.process(post)

            buffer.close()
            out_conn.send(decoder_state)

    @staticmethod
    def mix_to_stereo_worker(
        expander_l_in_conn, expander_r_in_conn, out_conn, peak_gain, sample_rate
    ):
        setproctitle(current_process().name)
        while True:
            while True:
                try:
                    l_decoder_state = expander_l_in_conn.recv()
                    break
                except InterruptedError:
                    pass
                except EOFError:
                    return

            while True:
                try:
                    r_decoder_state = expander_r_in_conn.recv()
                    break
                except InterruptedError:
                    pass
                except EOFError:
                    return

            assert (
                l_decoder_state.block_num == r_decoder_state.block_num
            ), "Noise reduction processes are out of sync! Channels will be out of sync."

            decoder_state = l_decoder_state
            buffer = PostProcessorSharedMemory(decoder_state)
            l = buffer.get_post_left()
            r = buffer.get_post_right()
            stereo = buffer.get_stereo()

            max_gain_left, max_gain_right = PostProcessor.stereo_interleave(
                l, r, stereo, sample_rate, decoder_state.block_num == 0
            )

            with peak_gain.get_lock():
                if peak_gain.left < max_gain_left:
                    peak_gain.left = max_gain_left

                if peak_gain.right < max_gain_right:
                    peak_gain.right = max_gain_right

            buffer.close()
            out_conn.send(decoder_state)

    @staticmethod
    @njit(
        numba.types.UniTuple(numba.types.float32, 2)(
            NumbaAudioArray,
            NumbaAudioArray,
            NumbaAudioArray,
            numba.types.int32,
            numba.types.bool_,
        ),
        cache=True,
        fastmath=True,
        nogil=True,
    )
    def stereo_interleave(
        audioL: np.array,
        audioR: np.array,
        stereo: np.array,
        sample_rate: int,
        is_first_block: bool,
    ) -> int:
        max_gain_left = 0
        max_gain_right = 0
        start_sample = 0

        # mute the spike that occurs during noise reduction
        if is_first_block:
            trim_samples = int(0.0015 * sample_rate)
            start_sample = trim_samples
            for i in range(trim_samples):
                stereo[i * 2] = 0
                stereo[i * 2 + 1] = 0

        channel_length = len(audioL)
        for i in range(start_sample, channel_length):
            audioLSample = audioL[i]
            stereo[i * 2] = audioLSample
            gain = abs(audioLSample)
            if gain > max_gain_left:
                max_gain_left = gain

            audioRSample = audioR[i]
            stereo[i * 2 + 1] = audioRSample
            gain = abs(audioRSample)
            if gain > max_gain_right:
                max_gain_right = gain

        return max_gain_left, max_gain_right

    @staticmethod
    def block_sorter_worker(
        decoder_out_queue,
        decoder_shared_memory_idle_queue,
        blocks_enqueued,
        post_processor_shared_memory_idle_queue,
        l_tx,
        r_tx,
    ):
        setproctitle(current_process().name)
        next_block = 0
        last_block_submitted = -1
        block_queue = []

        done = False
        while not done:
            while True:
                try:
                    in_decoder_state = decoder_out_queue.get()
                    break
                except InterruptedError:
                    pass
                except EOFError:
                    return

            buffer = DecoderSharedMemory(in_decoder_state)

            in_preL_buffer = buffer.get_pre_left()
            in_preR_buffer = buffer.get_pre_right()

            in_preL = np.empty(in_decoder_state.block_audio_final_len, dtype=REAL_DTYPE, order="C")
            in_preR = np.empty(in_decoder_state.block_audio_final_len, dtype=REAL_DTYPE, order="C")

            in_preL[:] = in_preL_buffer[: len(in_preL)]
            in_preR[:] = in_preR_buffer[: len(in_preR)]

            buffer.close()

            decoder_shared_memory_idle_queue.put((in_decoder_state.name, in_decoder_state.numa_node))
            with blocks_enqueued.get_lock():
                blocks_enqueued.value -= 1

            # blocks are received from the decoder processes out of order
            # gather them into ordered chunk and process sequentially
            assert (
                last_block_submitted < in_decoder_state.block_num
            ), f"Warning, block was repeated, got {in_decoder_state.block_num}, already processed {last_block_submitted}"
            block_queue.append((in_decoder_state, in_preL, in_preR))

            if in_decoder_state.block_num == next_block:
                # process queued data in order of block number
                block_queue.sort(key=lambda x: x[0].block_num)

                # enqueue the blocks in order
                while len(block_queue) > 0 and (
                    block_queue[0][0].block_num <= next_block
                ):
                    while True:
                        try:
                            name, numa_node = post_processor_shared_memory_idle_queue.get()
                            break
                        except InterruptedError:
                            pass
                        except EOFError:
                            return

                    decoder_state, preL, preR = block_queue.pop(0)

                    decoder_state.name = name
                    buffer = PostProcessorSharedMemory(decoder_state)

                    post_processor_preL = buffer.get_pre_left()
                    post_processor_preL[:] = preL[: len(post_processor_preL)]
                    post_processor_preR = buffer.get_pre_right()
                    post_processor_preR[:] = preR[: len(post_processor_preR)]

                    l_tx.send((decoder_state, 0))
                    r_tx.send((decoder_state, 1))

                    next_block += 1
                    last_block_submitted = decoder_state.block_num
                    done = decoder_state.is_last_block

    def close(self):
        cleanup_process(self.block_sorter_process)
        cleanup_process(self.dc_blocker_worker_l)
        cleanup_process(self.dc_blocker_worker_r)
        cleanup_process(self.spectral_nr_worker_l)
        cleanup_process(self.spectral_nr_worker_r)
        cleanup_process(self.expander_worker_l)
        cleanup_process(self.expander_worker_r)
        cleanup_process(self.mix_to_stereo_worker_process)
