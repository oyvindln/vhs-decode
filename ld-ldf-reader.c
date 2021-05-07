/*
 * Decode 16-bit samples from a compressed file using ffmpeg's libraries.
 *
 * Copyright (c) 2019-2021 Chad Page
 * Copyright (c) 2020-2021 Adam Sampson
 *
 * Adapted from ffmpeg's doc/examples/demuxing_decoding.c, which is:
 * Copyright (c) 2012 Stefano Sabatini
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>

static AVFormatContext *fmt_ctx = NULL;
static AVCodecContext *audio_dec_ctx;
static AVStream *audio_stream = NULL;
static const char *src_filename = NULL;

static int audio_stream_idx = -1;
static AVFrame *frame = NULL;
static AVPacket *pkt = NULL;

static uint64_t seekto = 0;

static int decode_packet(AVCodecContext *dec, const AVPacket *pkt)
{
    int ret = 0;

    // submit the packet to the decoder
    ret = avcodec_send_packet(dec, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error submitting a packet for decoding (%s)\n", av_err2str(ret));
        return ret;
    }

    // get all the available frames from the decoder
    while (ret >= 0) {
        ret = avcodec_receive_frame(dec, frame);
        if (ret < 0) {
            // those two return values are special and mean there is no output
            // frame available, but there were no errors during decoding
            if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
                return 0;

            fprintf(stderr, "Error during decoding (%s)\n", av_err2str(ret));
            return ret;
        }

        // If we haven't reached the start position, don't output anything
        if ((frame->pts + frame->nb_samples) < seekto) {
            av_frame_unref(frame);
            continue;
        }

        // Write the raw audio data samples to stdout, skipping any data that's
        // before the start position
        int64_t offset = FFMIN((seekto - frame->pts), 0);
        size_t unpadded_linesize = frame->nb_samples * av_get_bytes_per_sample(frame->format);
        size_t rv = write(1, frame->extended_data[0] + (offset * av_get_bytes_per_sample(frame->format)), unpadded_linesize);
        if (rv != unpadded_linesize) {
            fprintf(stderr, "write error %ld", offset);
            return -1;
        }

        av_frame_unref(frame);
    }

    return 0;
}

static int open_codec_context(int *stream_idx,
                              AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx, enum AVMediaType type)
{
    int ret, stream_index;
    AVStream *st;
    const AVCodec *dec = NULL;
    AVDictionary *opts = NULL;

    ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);

    if (ret < 0) {
        fprintf(stderr, "Could not find %s stream in input file '%s'\n",
                av_get_media_type_string(type), src_filename);
        return ret;
    } else {
        stream_index = ret;
        st = fmt_ctx->streams[stream_index];

        /* find decoder for the stream */
        dec = avcodec_find_decoder(st->codecpar->codec_id);
        if (!dec) {
            fprintf(stderr, "Failed to find %s codec\n",
                    av_get_media_type_string(type));
            return AVERROR(EINVAL);
        }

        /* Allocate a codec context for the decoder */
        *dec_ctx = avcodec_alloc_context3(dec);
        if (!*dec_ctx) {
            fprintf(stderr, "Failed to allocate the %s codec context\n",
                    av_get_media_type_string(type));
            return AVERROR(ENOMEM);
        }

        /* Copy codec parameters from input stream to output codec context */
        if ((ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar)) < 0) {
            fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
                    av_get_media_type_string(type));
            return ret;
        }

        /* Init the decoder */
        if ((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0) {
            fprintf(stderr, "Failed to open %s codec\n",
                    av_get_media_type_string(type));
            return ret;
        }
        *stream_idx = stream_index;
    }

    return 0;
}

int main (int argc, char **argv)
{
    int ret = 0;

    if (argc != 2 && argc != 3) {
        fprintf(stderr, "usage: %s input_file [start_offset_in_samples]\n",
                argv[0]);
        exit(1);
    }

    src_filename = argv[1];
    if (argc >= 3) {
        seekto = atoll(argv[2]);
    }

#if LIBAVFORMAT_VERSION_MAJOR < 59
    av_register_all();
#else
    // Later versions of ffmpeg register all codecs automatically
#endif

    /* open input file, and allocate format context */
    if ((ret = avformat_open_input(&fmt_ctx, src_filename, NULL, NULL)) < 0) {
        if (!strcmp(src_filename, "--help") || !strcmp(src_filename, "-h")) {
        	fprintf(stderr, "%s: Extract 16-bit data from .ldf (.oga compressed) files\n", argv[0]);
        	fprintf(stderr, "usage: %s [filename] [seek location]\n", argv[0]);
        	fprintf(stderr, "(output is streamed to standard output)\n");
        	exit(1);
	}
        fprintf(stderr, "Could not open source file %s %x\n", src_filename, ret);
        exit(1);
    }

    /* retrieve stream information */
    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
        fprintf(stderr, "Could not find stream information\n");
        exit(1);
    }

    if (open_codec_context(&audio_stream_idx, &audio_dec_ctx, fmt_ctx, AVMEDIA_TYPE_AUDIO) >= 0) {
        audio_stream = fmt_ctx->streams[audio_stream_idx];
    }

    /* dump input information to stderr */
    // av_dump_format(fmt_ctx, 0, src_filename, 0);

    if (!audio_stream) {
        fprintf(stderr, "Could not find audio stream in the input, aborting\n");
        ret = 1;
        goto end;
    }

    fprintf(stderr, "RATE:%d\n", audio_dec_ctx->sample_rate);
    // From fmt_ctx, this is the approximate length in ms.  (divide by 1000 for actual time)
    fprintf(stderr, "DURATION:%ld\n", fmt_ctx->duration);

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate frame\n");
        ret = AVERROR(ENOMEM);
        goto end;
    }

    pkt = av_packet_alloc();
    if (!pkt) {
        fprintf(stderr, "Could not allocate packet\n");
        ret = AVERROR(ENOMEM);
        goto end;
    }

    if (seekto) {
        uint64_t seeksec = seekto / audio_dec_ctx->sample_rate;

        avformat_seek_file(fmt_ctx, -1, (seeksec - 1) * 1000000, seeksec * 1000000, seeksec * 1000000, AVSEEK_FLAG_ANY);
    }

    /* read frames from the file */
    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == audio_stream_idx)
            ret = decode_packet(audio_dec_ctx, pkt);
        av_packet_unref(pkt);
        if (ret < 0)
            break;
    }

    /* flush the decoder */
    decode_packet(audio_dec_ctx, NULL);

end:
    avcodec_free_context(&audio_dec_ctx);
    avformat_close_input(&fmt_ctx);
    av_packet_free(&pkt);
    av_frame_free(&frame);

    return ret < 0;
}
