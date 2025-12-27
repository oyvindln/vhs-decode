use std::io::{self};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(target_os = "windows")]
use std::mem;


fn page_size() -> usize {
    #[cfg(not(target_os = "windows"))]
    unsafe {
        let ps = libc::sysconf(libc::_SC_PAGESIZE);
        if ps <= 0 { 4096 } else { ps as usize }
    }

    #[cfg(target_os = "windows")]
    unsafe {
        use winapi::um::sysinfoapi::{GetSystemInfo, SYSTEM_INFO};
        let mut info: SYSTEM_INFO = std::mem::zeroed();
        GetSystemInfo(&mut info);
        let ps = info.dwPageSize as usize;
        if ps == 0 { 4096 } else { ps }
    }
}

// Unix-specific sys module (Linux, macOS, BSD)
#[cfg(not(target_os = "windows"))]
mod sys {
    pub type RawFd = std::os::unix::io::RawFd;

    pub fn memfd_create(name: &str, flags: u32) -> std::io::Result<RawFd> {
        // On Linux: use memfd_create
        #[cfg(target_os = "linux")]
        {
            let c_name = std::ffi::CString::new(name).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid name"))?;
            unsafe {
                let fd = libc::syscall(libc::SYS_memfd_create, c_name.as_ptr(), flags as usize);
                if fd < 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(fd as RawFd)
                }
            }
        }

        // On macOS and BSD: use shm_open
        #[cfg(any(target_os = "macos", target_os = "freebsd", target_os = "openbsd", target_os = "netbsd"))]
        {
            let c_name = std::ffi::CString::new(name).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid name"))?;
            unsafe {
                let fd = libc::shm_open(c_name.as_ptr(), libc::O_RDWR | libc::O_CREAT | libc::O_EXCL, 0o600);
                if fd < 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    // Unlink immediately to avoid leaks
                    let _ = libc::shm_unlink(c_name.as_ptr());
                    Ok(fd)
                }
            }
        }
    }

    pub fn ftruncate(fd: RawFd, len: usize) ->  std::io::Result<()> {
        unsafe {
            if libc::ftruncate(fd, len as i64) < 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        }
    }

    pub fn mmap(
        addr: *mut u8,
        len: usize,
        prot: i32,
        flags: i32,
        fd: RawFd,
        offset: usize,
    ) ->  std::io::Result<*mut u8> {
        unsafe {
            let ptr = libc::mmap(addr as *mut libc::c_void, len, prot, flags, fd, offset as i64);
            if ptr == libc::MAP_FAILED {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(ptr as *mut u8)
            }
        }
    }

    pub fn munmap(addr: *mut u8, len: usize) ->  std::io::Result<()> {
        unsafe {
            if libc::munmap(addr as *mut libc::c_void, len) < 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        }
    }
}

// Windows-specific sys module
#[cfg(target_os = "windows")]
mod sys {
    pub type RawFd = winapi::shared::ntdef::HANDLE;

    pub fn page_granularity() -> usize {
        unsafe {
            let mut info: winapi::um::sysinfoapi::SYSTEM_INFO = std::mem::zeroed();
            winapi::um::sysinfoapi::GetSystemInfo(&mut info);
            info.dwAllocationGranularity as usize
        }
    }

    // Reserve 2*size with placeholders, release first half placeholder,
    // return base address (maparea).
    pub unsafe fn reserve_placeholder_2x(size: usize) -> std::io::Result<*mut u8> {
        use std::ptr;

        let maparea = winapi::um::memoryapi::VirtualAlloc2(
            ptr::null_mut(),
            ptr::null_mut(),
            2 * size,
            winapi::um::winnt::MEM_RESERVE | winapi::um::winnt::MEM_RESERVE_PLACEHOLDER,
            winapi::um::winnt::PAGE_NOACCESS,
            ptr::null_mut(),
            0,
        );

        if maparea.is_null() {
            return Err(std::io::Error::last_os_error());
        }

        let ok = winapi::um::memoryapi::VirtualFree(
            maparea,
            size,
            winapi::um::winnt::MEM_RELEASE | winapi::um::winnt::MEM_PRESERVE_PLACEHOLDER,
        );
        if ok == 0 {
            let _ = winapi::um::memoryapi::VirtualFree(maparea, 0, winapi::um::winnt::MEM_RELEASE);
            return Err(std::io::Error::last_os_error());
        }

        Ok(maparea as *mut u8)
    }

    pub unsafe fn release_region(addr: *mut u8) {
        let _ = winapi::um::memoryapi::VirtualFree(addr as *mut _, 0, winapi::um::winnt::MEM_RELEASE);
    }

    pub unsafe fn create_mapping(size: usize) -> std::io::Result<RawFd> {
        use std::ptr;
        let h = winapi::um::memoryapi::CreateFileMappingW(
            winapi::um::handleapi::INVALID_HANDLE_VALUE,
            ptr::null_mut(),
            winapi::um::winnt::PAGE_READWRITE,
            0,
            size as u32, // sizes >4GB would need the high DWORD too
            ptr::null(),
        );
        if h.is_null() {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(h)
        }
    }

    pub unsafe fn close_handle(h: RawFd) {
        let _ = winapi::um::handleapi::CloseHandle(h);
    }

    pub unsafe fn map_view_replace_placeholder(
        h: RawFd,
        base: *mut u8,
        size: usize,
    ) -> std::io::Result<*mut u8> {
        use std::ptr;

        let p = winapi::um::memoryapi::MapViewOfFile3(
            h,
            ptr::null_mut(),
            base as *mut _,
            0,
            size,
            winapi::um::winnt::MEM_REPLACE_PLACEHOLDER,
            winapi::um::winnt::PAGE_READWRITE,
            ptr::null_mut(),
            0,
        );
        if p.is_null() {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(p as *mut u8)
        }
    }

    pub unsafe fn unmap_view(addr: *mut u8) -> std::io::Result<()> {
        let ok = winapi::um::memoryapi::UnmapViewOfFile(addr as *const _);
        if ok == 0 { Err(std::io::Error::last_os_error()) } else { Ok(()) }
    }
}

// RingBuffer: single-producer, single-consumer, zero-copy, no locks
pub struct RingBuffer {
    buffer: *mut u8,
    buffer2: *mut u8,
    buffer_size: usize,
    #[cfg(not(target_os="windows"))]
    fd: sys::RawFd,
    head: AtomicUsize,
    tail: AtomicUsize,
}

// Safety: we're using raw pointers and mmap, but only one instance per buffer
// and we ensure no aliasing via the double-mapping trick.
unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    pub fn new(name: &str, size: usize) -> io::Result<Self> {
        let page_size = page_size();
        let size = (size + page_size - 1) / page_size * page_size;
        if size == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "size must be positive"));
        }

        #[cfg(target_os="windows")]
        unsafe {
            let maparea = sys::reserve_placeholder_2x(size)?;

            let h = match sys::create_mapping(size) {
                Ok(h) => h,
                Err(e) => {
                    sys::release_region(maparea);
                    sys::release_region(maparea.add(size));
                    return Err(e);
                }
            };

            let buffer = match sys::map_view_replace_placeholder(h, maparea, size) {
                Ok(p) => p,
                Err(e) => {
                    sys::close_handle(h);
                    sys::release_region(maparea);
                    sys::release_region(maparea.add(size));
                    return Err(e);
                }
            };

            let buffer2 = match sys::map_view_replace_placeholder(h, maparea.add(size), size) {
                Ok(p) => p,
                Err(e) => {
                    let _ = sys::unmap_view(buffer);
                    sys::close_handle(h);
                    sys::release_region(maparea.add(size));
                    return Err(e);
                }
            };

            sys::close_handle(h);

            return Ok(RingBuffer {
                buffer,
                buffer2,
                buffer_size: size,
                fd: std::ptr::null_mut(), // not used on Windows; keep field, or cfg it out
                head: std::sync::atomic::AtomicUsize::new(0),
                tail: std::sync::atomic::AtomicUsize::new(0),
            });
        }


        #[cfg(not(target_os="windows"))]
        {
            let fd = sys::memfd_create(name, 0)?;
            sys::ftruncate(fd, size)?;

            // Allocate two pages of virtual memory (we'll map the same fd twice)
            let total_size = 2 * size;
            let buffer = sys::mmap(
                ptr::null_mut(),
                total_size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )?;

            // Map the buffer at the first half
            sys::mmap(
                buffer,
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                0,
            )?;

            // Map the buffer again at the second half
            sys::mmap(
                unsafe { buffer.add(size) },
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                0,
            )?;

            return Ok(RingBuffer {
                buffer,
                buffer2: unsafe { buffer.add(size) },
                buffer_size: size,
                fd,
                head: AtomicUsize::new(0),
                tail: AtomicUsize::new(0),
            })
        }
    }

    // Returns a pointer to write into (no copy)
    pub fn write_ptr(&self, size: usize) -> Option<*mut u8> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);

        // used bytes in ring
        let used = tail.wrapping_sub(head);
        if self.buffer_size - used < size {
            return None;
        }

        // pointer uses modulo (works with double mapping)
        let off = tail % self.buffer_size;
        unsafe { Some(self.buffer.add(off)) }
    }

    // Finish writing `size` bytes
    pub fn write_finished(&self, size: usize) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);

        let used = tail.wrapping_sub(head);
        if self.buffer_size - used < size {
            return false;
        }

        self.tail.store(tail.wrapping_add(size), Ordering::Release);
        true
    }

    // Returns a pointer to read from (no copy)
    pub fn read_ptr(&self, size: usize) -> Option<*const u8> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);

        let used = tail.wrapping_sub(head);
        if used < size {
            return None;
        }
        let off = head % self.buffer_size;
        unsafe { Some(self.buffer.add(off) as *const u8) }
    }

    // Finish reading `size` bytes
    pub fn read_finished(&self, size: usize) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);

        let used = tail.wrapping_sub(head);
        if used < size {
            return false;
        }

        self.head.store(head.wrapping_add(size), Ordering::Release);
        true
    }

    pub fn reset(&self) {
        self.head.store(0, Ordering::Release);
        self.tail.store(0, Ordering::Release);
    }

    // Get current head and tail (for debugging or monitoring)
    pub fn head(&self) -> usize {
        self.head.load(Ordering::Acquire)
    }

    pub fn tail(&self) -> usize {
        self.tail.load(Ordering::Acquire)
    }

    pub fn capacity(&self) -> usize {
        self.buffer_size
    }

    pub fn available_read(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    pub fn available_write(&self) -> usize {
        self.buffer_size - self.available_read()
    }
}

impl Drop for RingBuffer {
    fn drop(&mut self) {
        unsafe {
            #[cfg(target_os="windows")]
            {
                let _ = sys::unmap_view(self.buffer);
                let _ = sys::unmap_view(self.buffer2);
            }
            #[cfg(not(target_os="windows"))]
            {
                let _ = sys::munmap(self.buffer, 2 * self.buffer_size);
                let _ = libc::close(self.fd);
            }
        }
    }
}

// Helper: create a ring buffer with a name and size
pub fn create_ringbuffer(name: &str, size: usize) -> io::Result<RingBuffer> {
    RingBuffer::new(name, size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn test_ringbuffer_randomized_spsc_10mib() {
        const RING_SIZE: usize = 1 * 1024 * 1024; // 1 MiB
        const TOTAL: usize = 10 * 1024 * 1024; // 10 MiB
        const MAX_CHUNK: usize = 100 * 1024; // 100 KiB

        // Generate 10 MiB of random data into a "static" (fixed-size) buffer.
        let mut src = vec![0u8; TOTAL];
        let mut rng = StdRng::seed_from_u64(0xD1CE_F00D_CAFE_BABE);
        rng.fill(&mut src[..]);

        // Destination buffer ("static"/fixed-size).
        let mut dst = vec![0u8; TOTAL];

        let rb = Arc::new(create_ringbuffer("test_ringbuffer_rand", RING_SIZE).unwrap());

        // Writer thread: writes TOTAL bytes from src into rb in random chunks [1..=MAX_CHUNK].
        let rb_w = Arc::clone(&rb);
        let src_w = Arc::new(src);
        let src_w2 = Arc::clone(&src_w);

        let writer = thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(0x1234_5678_9ABC_DEF0);
            let mut written = 0usize;

            while written < TOTAL {
                let remaining = TOTAL - written;
                let want = rng.random_range(1..=MAX_CHUNK.min(remaining));

                if let Some(wptr) = rb_w.write_ptr(want) {
                    unsafe {
                        std::ptr::copy_nonoverlapping(src_w2.as_ptr().add(written), wptr, want);
                    }
                    assert!(rb_w.write_finished(want));
                    written += want;
                } else {
                    thread::sleep(Duration::from_micros(50));
                }
            }
        });

        // Reader (in test thread): reads TOTAL bytes into dst in random chunks [1..=MAX_CHUNK].
        let mut rng_r = StdRng::seed_from_u64(0x0BAD_F00D_DEAD_BEEF);
        let mut read = 0usize;

        while read < TOTAL {
            let remaining = TOTAL - read;
            let want = rng_r.random_range(1..=MAX_CHUNK.min(remaining));

            if let Some(rptr) = rb.read_ptr(want) {
                unsafe {
                    std::ptr::copy_nonoverlapping(rptr, dst.as_mut_ptr().add(read), want);
                }
                assert!(rb.read_finished(want));
                read += want;
            } else {
                thread::sleep(Duration::from_micros(50));
            }
        }

        // Ensure writer finished and compare.
        writer.join().expect("writer thread panicked");
        assert_eq!(&dst[..], &src_w[..]);
    }
}
