use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::slice;
use std::sync::{
    atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;
use std::ffi::CStr;
use libflac_sys::*;

use crate::ringbuffer::RingBuffer;

type ConvFn = fn(input: &[u8], output: &mut [f32], len: usize, shift: i32);

const BUFFER_SAMPLE_SIZE : u64 = 65536 * 512;
const BUFFER_SAMPLES_KEEP : u64 = 65536;

pub enum InputFileType {
    FLAC,
    S16,
    U8,
}

fn conv_s16_to_f32(input: &[u8], output: &mut [f32], len: usize, _shift: i32) {
    let in16: &[i16] = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const i16, len) };
    for i in 0..len {
        output[i] = in16[i] as f32;
    }
}

fn conv_u8_to_f32(input: &[u8], output: &mut [f32], len: usize, _shift: i32) {
    for i in 0..len {
        output[i] = (input[i] as f32 - 128.0) * 256.0;
    }
}

// for flac sources, shift for non 16-bit sources
fn conv_s32_to_f32(input: &[u8], output: &mut [f32], len: usize, shift: i32) {
    let in32: &[i32] = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const i32, len) };
    if shift == 0 {
        for i in 0..len {
            output[i] = in32[i] as f32;
        }
    }
    else if shift > 0 {
        for i in 0..len {
            output[i] = (in32[i] << shift) as f32;
        }
    }
    else {
        let scale = 2.0f32.powi(-shift);
        let inv_scale = 1.0f32 / scale;

        for i in 0..len {
            output[i] = (in32[i] as f32) * inv_scale;
        }
    }
}

#[repr(C)]
pub struct InputFile {
    // The real context shared with the reader thread:
    inner: Arc<InputFileInner>,

    // JoinHandle is not shared; owned by the handle wrapper returned to the caller:
    reader_thread: Option<thread::JoinHandle<()>>,
}

#[repr(C)]
pub struct InputFileInner {
    pub filename: PathBuf,
    pub rb: Arc<RingBuffer>,
    pub input_type: InputFileType,

    pub conv_func: ConvFn,

    // Filled from STREAMINFO for FLAC
    pub sample_rate: AtomicU64,
    pub bits_per_sample: AtomicU32,
    pub shift: AtomicI32,
    pub bytes_per_sample: u32,

    pub read_pos: AtomicU64,
    pub seek_sample: AtomicU64,

    // control/state
    pub resample: bool,
    pub seek: AtomicBool,
    pub seek_clear: AtomicBool,
    pub metadata_read: AtomicBool,
    pub exit: AtomicBool,
    pub finished: AtomicBool,
    pub error: AtomicBool,
}

impl InputFile {
    pub fn new(
        filename: impl Into<PathBuf>,
        sample_rate: u64,
        resample: bool,
        input_type: InputFileType,
    ) -> io::Result<Self> {
        let inner = Arc::new(InputFileInner {
            filename: filename.into(),
            rb: Arc::new(
                RingBuffer::new(
                    "ringbuffer_inputfile",
                    (BUFFER_SAMPLE_SIZE * 4).try_into().unwrap(),
                )
                .unwrap(),
            ),
            conv_func: match input_type {
                InputFileType::S16 => conv_s16_to_f32,
                InputFileType::U8 => conv_u8_to_f32,
                InputFileType::FLAC => conv_s32_to_f32,
            },
            sample_rate: AtomicU64::new(sample_rate),
            bits_per_sample: AtomicU32::new(0),
            shift: AtomicI32::new(0),
            bytes_per_sample: match input_type {
                InputFileType::S16 => 2,
                InputFileType::U8  => 1,
                InputFileType::FLAC => 0, // set later from STREAMINFO if needed
            },
            input_type: input_type,
            read_pos: AtomicU64::new(0),
            seek_sample: AtomicU64::new(0),
            resample: resample,
            seek: AtomicBool::new(false),
            seek_clear: AtomicBool::new(false),
            metadata_read: AtomicBool::new(false),
            exit: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            error: AtomicBool::new(false),
        });

        let inner2 = Arc::clone(&inner);
        let reader_thread = match inner.input_type {
            InputFileType::S16 | InputFileType::U8 => {
                thread::spawn(move || raw_reader_thread_main(inner2))
            }
            InputFileType::FLAC => thread::spawn(move || flac_reader_thread_main(inner2)),
        };

        Ok(Self {
            inner,
            reader_thread: Some(reader_thread),
        })
    }

    pub fn read(&mut self, buffer: &mut [f32], pos: u64, _len: usize) -> usize {
        if self.inner.error.load(Ordering::Acquire) || self.inner.exit.load(Ordering::Acquire) {
            return 0;
        }
        let mut len = _len;
        let mut rp = self.inner.read_pos.load(Ordering::Acquire);
        if pos < rp || pos > rp + BUFFER_SAMPLE_SIZE { // we need to seek, should not happen usually
            if pos > BUFFER_SAMPLES_KEEP {
                self.inner.seek_sample.store(pos - BUFFER_SAMPLES_KEEP, Ordering::Relaxed);
            }
            else {
                self.inner.seek_sample.store(0, Ordering::Relaxed);
            }
            self.inner.seek.store(true, Ordering::Release);
            self.inner.seek_clear.store(true, Ordering::Release);
            loop {
                if self.inner.error.load(Ordering::Acquire) || self.inner.exit.load(Ordering::Acquire) {
                    return 0;
                }
                if self.inner.seek.load(Ordering::Acquire) == false {
                    rp = self.inner.read_pos.load(Ordering::Acquire);
                    break;
                }
                thread::sleep(Duration::from_micros(500));
            }
        }
        // we always leave BUFFER_SAMPLES_KEEP samples in the ringbuffer to prevent back seeks
        if pos > (rp + BUFFER_SAMPLES_KEEP) {
            let consume = pos - (rp + BUFFER_SAMPLES_KEEP);
            loop {
                if self.inner.rb.read_finished((consume*4).try_into().unwrap()) {
                    rp += consume;
                    self.inner.read_pos.store(rp, Ordering::Relaxed);
                    break;
                }
                if self.inner.error.load(Ordering::Acquire) || self.inner.exit.load(Ordering::Acquire) {
                    return 0;
                } else {
                    thread::sleep(Duration::from_micros(50));
                }
            }
        }
        loop {
            if let Some(rptr) = self.inner.rb.read_ptr((((len as u64) + (pos - rp))*4).try_into().unwrap()) {
                unsafe { std::ptr::copy_nonoverlapping(rptr.add(((pos - rp)*4) as usize), buffer.as_mut_ptr() as *mut u8, (len*4).try_into().unwrap()); }
                break;
            }
            if self.inner.error.load(Ordering::Acquire) || self.inner.exit.load(Ordering::Acquire) {
                return 0;
            }
            if self.inner.finished.load(Ordering::Acquire) {
                let avail = self.inner.rb.available_read() / 4;
                let offset_samp = (pos - rp) as usize;

                if offset_samp >= avail {
                    return 0;
                }

                let remaining = avail - offset_samp;
                len = len.min(remaining);
            } else {
                thread::sleep(Duration::from_micros(50));
            }
        }
        len
    }

    pub fn is_error(&mut self) -> bool {
        return self.inner.error.load(Ordering::Acquire);
    }

    pub fn get_samplerate(&mut self) -> i64 {
        loop {
            if self.inner.error.load(Ordering::Acquire) || self.inner.exit.load(Ordering::Acquire) {
                return -1;
            }
            if self.inner.metadata_read.load(Ordering::Acquire) {
                return self.inner.sample_rate.load(Ordering::Acquire) as i64;
            }
        }
    }

    pub fn close(&mut self) -> bool {
        self.inner.exit.store(true, Ordering::Release);
        let jr = match self.reader_thread.take() {
            Some(h) => h.join().is_ok(),
            None => false, // already joined / never started
        };
        return (!self.inner.error.load(Ordering::Acquire)) && jr;
    }
}

fn fail(ctx: &Arc<InputFileInner>) {
    ctx.error.store(true, Ordering::Release);
}

unsafe extern "C" fn flac_error_cb(
    _decoder: *const FLAC__StreamDecoder,
    status: FLAC__StreamDecoderErrorStatus,
    c_ctx: *mut std::ffi::c_void,
) {
    let ctx = &*(c_ctx as *const Arc<InputFileInner>);
    let s = CStr::from_ptr(FLAC__StreamDecoderErrorStatusString[status as usize]);
    eprintln!("FLAC decoding error: {}", s.to_string_lossy());
    ctx.error.store(true, Ordering::Release);
}

unsafe extern "C" fn flac_metadata_cb(
    _decoder: *const FLAC__StreamDecoder,
    metadata: *const FLAC__StreamMetadata,
    c_ctx: *mut std::ffi::c_void,
) {
    let ctx = &*(c_ctx as *const Arc<InputFileInner>);
    if metadata.is_null() {
        ctx.error.store(true, Ordering::Release);
        return;
    }
    let md = &*metadata;

    if md.type_ != FLAC__METADATA_TYPE_STREAMINFO {
        return;
    }

    let si = md.data.stream_info;

    if si.channels != 1 {
        ctx.error.store(true, Ordering::Release);
        return;
    }

    let sample_rate = ctx.sample_rate.load(Ordering::Acquire);

    if sample_rate == 0 { // could have been overwritten manually
        ctx.sample_rate.store((si.sample_rate as u64) * 1000, Ordering::Release);
    }

    ctx.bits_per_sample.store(si.bits_per_sample, Ordering::Release);
    ctx.shift.store(16 as i32 - (si.bits_per_sample as i32), Ordering::Release);
    ctx.metadata_read.store(true, Ordering::Release);
}

unsafe extern "C" fn flac_data_cb(
    _decoder: *const FLAC__StreamDecoder,
    frame: *const FLAC__Frame,
    buffer: *const *const FLAC__int32,
    c_ctx: *mut std::ffi::c_void,
) -> FLAC__StreamDecoderWriteStatus {
    let ctx = &*(c_ctx as *const Arc<InputFileInner>);

    if ctx.exit.load(Ordering::Acquire) || ctx.error.load(Ordering::Acquire) {
        return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
    }

    let blocksize = (*frame).header.blocksize as usize;
    let total_bytes = blocksize * std::mem::size_of::<i32>();
    
    //let sample = (*frame).header.number.sample_number as u64;

    // channel 0 pointer (mono); libFLAC gives FLAC__int32 (usually i32)
    let src_i32 = *buffer; // *const FLAC__int32
    let src_u8 = src_i32 as *const u8;

    loop {
        if ctx.exit.load(Ordering::Acquire) || ctx.error.load(Ordering::Acquire) {
            return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
        }
        if ctx.seek_clear.load(Ordering::Acquire) {
            return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
        }
        if let Some(wptr) = ctx.rb.write_ptr(total_bytes) {
            (ctx.conv_func)(slice::from_raw_parts(src_u8, total_bytes as usize), slice::from_raw_parts_mut(wptr as *mut f32, blocksize as usize), blocksize, ctx.shift.load(Ordering::Relaxed));
            if !ctx.rb.write_finished(total_bytes) {
                ctx.error.store(true, Ordering::Release);
                return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
            }
            return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
        } else {
            thread::sleep(Duration::from_micros(1000));
        }
    }
}

fn flac_reader_thread_main(ctx: Arc<InputFileInner>) {
    let mut f = match File::open(&ctx.filename) {
        Ok(f) => f,
        Err(_) => return fail(&ctx),
    };

    let mut magic = [0u8; 4];
    if f.read_exact(&mut magic).is_err() {
        return fail(&ctx);
    }

    drop(f);

    if magic != *b"fLaC" && magic != *b"OggS" {
        return fail(&ctx);
    }

    let decoder = unsafe { FLAC__stream_decoder_new() };
    if decoder.is_null() {
        return fail(&ctx);
    }

    // convert rust reference to c pointer
    let c_ctx: *mut std::ffi::c_void = Box::into_raw(Box::new(Arc::clone(&ctx))) as *mut _;

    let c_path = match std::ffi::CString::new(ctx.filename.to_string_lossy().as_bytes()) {
        Ok(s) => s,
        Err(_) => {
            unsafe { cleanup_flac_decoder(decoder, c_ctx) };
            return fail(&ctx);
        }
    };

    let init_status = if magic == *b"fLaC" {
        unsafe { FLAC__stream_decoder_init_file(
            decoder,
            c_path.as_ptr(),
            Some(flac_data_cb),
            Some(flac_metadata_cb),
            Some(flac_error_cb),
            c_ctx,
        ) }
    } else {
        unsafe { FLAC__stream_decoder_init_ogg_file(
            decoder,
            c_path.as_ptr(),
            Some(flac_data_cb),
            Some(flac_metadata_cb),
            Some(flac_error_cb),
            c_ctx,
        ) }
    };

    if init_status != FLAC__STREAM_DECODER_INIT_STATUS_OK {
        unsafe { cleanup_flac_decoder(decoder, c_ctx) };
        return fail(&ctx);
    }

    if unsafe { FLAC__stream_decoder_process_until_end_of_metadata(decoder) } == 0 {
        unsafe { cleanup_flac_decoder(decoder, c_ctx) };
        return fail(&ctx);
    }

    if ctx.error.load(Ordering::Acquire) {
        unsafe { cleanup_flac_decoder(decoder, c_ctx) };
        return fail(&ctx);
    }

    while !ctx.exit.load(Ordering::Acquire) && !ctx.error.load(Ordering::Acquire) {
        if ctx.seek.load(Ordering::Acquire) {
            unsafe { FLAC__stream_decoder_flush(decoder) };
            ctx.rb.reset();
            ctx.seek_clear.store(false, Ordering::Release);
            if unsafe { FLAC__stream_decoder_seek_absolute(decoder, ctx.seek_sample.load(Ordering::Relaxed)) } == 0 {
                unsafe { cleanup_flac_decoder(decoder, c_ctx) };
                return fail(&ctx);
            }
            ctx.read_pos.store(ctx.seek_sample.load(Ordering::Relaxed), Ordering::Relaxed);
            ctx.finished.store(false, Ordering::Release); // if we were at end of stream we now seek back
            ctx.seek.store(false, Ordering::Release);
        }
        if ctx.finished.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_micros(50000));
            continue;
        }
        unsafe { _ = FLAC__stream_decoder_process_single(decoder) }
        let state = unsafe { FLAC__stream_decoder_get_state(decoder) };
        if state == FLAC__STREAM_DECODER_END_OF_STREAM || state == FLAC__STREAM_DECODER_ABORTED {
            ctx.finished.store(true, Ordering::Release);
            thread::sleep(Duration::from_micros(50000));
        }
        else if state == FLAC__STREAM_DECODER_OGG_ERROR || state == FLAC__STREAM_DECODER_MEMORY_ALLOCATION_ERROR || state == FLAC__STREAM_DECODER_SEEK_ERROR {
            ctx.error.store(true, Ordering::Release);
            break;
        }
    }
    unsafe {
        FLAC__stream_decoder_finish(decoder);
        cleanup_flac_decoder(decoder, c_ctx);
    }
}

fn raw_reader_thread_main(ctx: Arc<InputFileInner>) {
    let mut f = match File::open(&ctx.filename) {
        Ok(f) => f,
        Err(_) => return fail(&ctx),
    };

    let num_samples = 65536;
    let mut buf = vec![0u8; (num_samples * ctx.bytes_per_sample).try_into().unwrap()];

    ctx.metadata_read.store(true, Ordering::Release); // there is no metadata

    while !ctx.exit.load(Ordering::Acquire) && !ctx.error.load(Ordering::Acquire) {
        if ctx.seek.load(Ordering::Acquire) {
            if f.seek(SeekFrom::Start(((ctx.bytes_per_sample as u64)* ctx.seek_sample.load(Ordering::Relaxed)).into())).is_err() {
                return fail(&ctx);
            }
            ctx.rb.reset();
            ctx.read_pos.store(ctx.seek_sample.load(Ordering::Relaxed), Ordering::Relaxed);
            ctx.finished.store(false, Ordering::Release);
            ctx.seek_clear.store(false, Ordering::Release);
            ctx.seek.store(false, Ordering::Release);
        }

        let mut nread = match f.read(&mut buf) {
            Ok(0) => {
                ctx.finished.store(true, Ordering::Release);
                thread::sleep(Duration::from_micros(50_000));
                continue;
            },
            Ok(n) => n,
            Err(_) => return fail(&ctx),
        };

        if nread % (ctx.bytes_per_sample as usize) != 0 {
            if nread < (ctx.bytes_per_sample as usize) {
                ctx.finished.store(true, Ordering::Release);
                thread::sleep(Duration::from_micros(50_000));
                continue;
            }
            if f.seek(SeekFrom::Current((-((nread % (ctx.bytes_per_sample as usize)) as i64)).try_into().unwrap())).is_err() {
                return fail(&ctx);
            }
        }
        nread = nread / (ctx.bytes_per_sample as usize);

        loop {
            if ctx.exit.load(Ordering::Acquire) || ctx.error.load(Ordering::Acquire) || ctx.seek_clear.load(Ordering::Acquire){ 
                break;
            }
            if let Some(wptr) = ctx.rb.write_ptr((nread * 4).try_into().unwrap()) {
                unsafe { (ctx.conv_func)(&buf, slice::from_raw_parts_mut(wptr as *mut f32, nread as usize), nread, 0); }
                if !ctx.rb.write_finished(nread*4) {
                    return fail(&ctx);
                }
                break;
            } else {
                thread::sleep(Duration::from_micros(50));
            }
        }
    }
    drop (f);
}

unsafe fn cleanup_flac_decoder(decoder: *mut FLAC__StreamDecoder, c_ctx: *mut std::ffi::c_void) {
    FLAC__stream_decoder_delete(decoder);
    drop(Box::from_raw(c_ctx as *mut Arc<InputFileInner>));
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use std::cmp::min;

    // ---- Configure these for your repository ----
    // Paths are relative to crate root by default; adjust as needed.
    const RAW_S16_PATH: &str = "/home/stefan/575tsgWvbi_svidBO_svo2krec_pb788_1ufmodopa657_13+_ex49tt_g54cx1dxvx1_resample20msps.flac";
    const RAW_S16_TOTAL_SAMPLES: u64 = 2_722_966_541;

    const RAW_U8_PATH: &str = "/home/stefan/575tsgWvbi_svidBO_svo2krec_pb788_1ufmodopa657_13+_ex49tt_g54cx1dxvx1_resample20msps.flac";
    const RAW_U8_TOTAL_SAMPLES: u64 = 5_445_933_083;

    const FLAC_PATH: &str = "/home/stefan/575tsgWvbi_svidBO_svo2krec_pb788_1ufmodopa657_13+_ex49tt_g54cx1dxvx1_resample20msps.flac";
    const FLAC_TOTAL_SAMPLES: u64 = 7_732_034_130; //386_601_706;

    // Test parameters
    const NUM_RANGES: usize = 128;
    const MIN_LEN: usize = 1_000;
    const MAX_LEN: usize = 100_000;

    // Extra EOF-overrun test
    const EOF_REQ_LEN: usize = 50_000;
    const EOF_IN_BOUNDS: u64 = 30_000; // so 20k past EOF when len=50k

    #[derive(Clone, Debug)]
    struct RangeReq {
        pos: u64,
        len: usize,
    }

    fn gen_ranges(rng: &mut impl Rng, total_samples: u64) -> Vec<RangeReq> {
        assert!(total_samples > 0);
        let max_len = min(MAX_LEN as u64, total_samples) as usize;

        (0..NUM_RANGES)
            .map(|_| {
                let len = rng.random_range(MIN_LEN..=max_len);
                let max_pos = total_samples.saturating_sub(len as u64);
                let pos = if max_pos == 0 { 0 } else { rng.random_range(0..=max_pos) };
                RangeReq { pos, len }
            })
            .collect()
    }

    fn read_ranges_in_order(
        inp: &mut InputFile,
        ranges: &[RangeReq],
        order: &[usize],
        pass_name: &str,
    ) -> Vec<Vec<f32>> {
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); ranges.len()];

        for (k, &idx) in order.iter().enumerate() {
            let r = &ranges[idx];

            eprintln!(
                "[{pass_name}] request #{k:03} range_idx={idx:03} pos={} len={}",
                r.pos, r.len
            );

            let mut buf = vec![0.0f32; r.len];
            let got = inp.read(&mut buf, r.pos, r.len);

            // NEW: check error state right after each read
            if inp.is_error() {
                panic!(
                    "[{pass_name}] error state set right after read(): \
                     request #{k:03} range_idx={idx:03} pos={} len={} got={}",
                    r.pos, r.len, got
                );
            }

            eprintln!(
                "[{pass_name}]   -> got {} samples (requested {})",
                got, r.len
            );

            buf.truncate(got);
            out[idx] = buf;
        }

        out
    }

    fn preview(v: &[f32], n: usize) -> String {
        let n = n.min(v.len());
        let mut s = String::new();
        for (i, &x) in v.iter().take(n).enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("{:.1}(0x{:08x})", x, x.to_bits()));
        }
        s
    }

    fn assert_same_samples(ranges: &[RangeReq], a: &[Vec<f32>], b: &[Vec<f32>]) {
        assert_eq!(a.len(), b.len());

        for i in 0..a.len() {
            if a[i].len() != b[i].len() {
                eprintln!(
                    "LENGTH MISMATCH at range_idx={i:03} pos={} len={} : pass1 got {}, pass2 got {}",
                    ranges[i].pos,
                    ranges[i].len,
                    a[i].len(),
                    b[i].len()
                );
                eprintln!("pass1 head: {}", preview(&a[i], 8));
                eprintln!("pass2 head: {}", preview(&b[i], 8));
                panic!("range {i} length differs");
            }

            for j in 0..a[i].len() {
                let x = a[i][j];
                let y = b[i][j];
                if x.to_bits() != y.to_bits() {
                    eprintln!(
                        "DATA MISMATCH at range_idx={i:03} pos={} len={} sample_offset={} (abs_sample={})",
                        ranges[i].pos,
                        ranges[i].len,
                        j,
                        ranges[i].pos + j as u64
                    );
                    eprintln!("pass1[{j}] = {x} (0x{:08x})", x.to_bits());
                    eprintln!("pass2[{j}] = {y} (0x{:08x})", y.to_bits());

                    let start = j.saturating_sub(4);
                    let end = (j + 5).min(a[i].len());
                    eprintln!(
                        "pass1 window [{}..{}]: {}",
                        start,
                        end,
                        preview(&a[i][start..end], 16)
                    );
                    eprintln!(
                        "pass2 window [{}..{}]: {}",
                        start,
                        end,
                        preview(&b[i][start..end], 16)
                    );
                    panic!("mismatch at range {i}, sample {j}");
                }
            }
        }
    }

    fn run_random_repeated_read_test(
        path: &str,
        total_samples: u64,
        sample_rate: u64,
        input_type: InputFileType,
    ) {
        let mut rng = StdRng::seed_from_u64(0xC0FFEEu64);

        let mut ranges = gen_ranges(&mut rng, total_samples);

        // Add the extra EOF-overrun range as the last element (fixed index)
        assert!(total_samples >= EOF_IN_BOUNDS);
        let eof_pos = total_samples - EOF_IN_BOUNDS;
        ranges.push(RangeReq {
            pos: eof_pos,
            len: EOF_REQ_LEN,
        });
        let eof_idx = ranges.len() - 1;

        let mut order1: Vec<usize> = (0..ranges.len()).collect();
        order1.shuffle(&mut rng);
        order1.push(eof_idx);

        let mut order2 = order1.clone();
        order2.shuffle(&mut rng);

        eprintln!("=== test file: {path} ===");
        eprintln!("EOF overrun req: range_idx={eof_idx} pos={eof_pos} len={EOF_REQ_LEN} ({} past EOF)",
                  (eof_pos + EOF_REQ_LEN as u64).saturating_sub(total_samples));
        eprintln!("order1: {:?}", order1);
        eprintln!("order2: {:?}", order2);

        let mut inp = InputFile::new(path, sample_rate, false, input_type)
            .unwrap_or_else(|e| panic!("failed to open {path}: {e}"));

        let pass1 = read_ranges_in_order(&mut inp, &ranges, &order1, "pass1");
        let pass2 = read_ranges_in_order(&mut inp, &ranges, &order2, "pass2");

        assert_same_samples(&ranges, &pass1, &pass2);

        let ok = inp.close();
        eprintln!("close() -> {ok}");
        assert!(ok, "InputFile::close() reported failure");
    }

    #[test]
    fn random_reads_raw_s16_repeatable() {
        run_random_repeated_read_test(RAW_S16_PATH, RAW_S16_TOTAL_SAMPLES, 0, InputFileType::S16);
    }

    #[test]
    fn random_reads_raw_u8_repeatable() {
        run_random_repeated_read_test(RAW_U8_PATH, RAW_U8_TOTAL_SAMPLES, 0, InputFileType::U8);
    }

    #[test]
    fn random_reads_flac_repeatable() {
        run_random_repeated_read_test(FLAC_PATH, FLAC_TOTAL_SAMPLES, 0, InputFileType::FLAC);
    }
}