// #![windows_subsystem = "windows"]
#![allow(dead_code)]
pub use color_eyre::eyre::eyre;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::{
    f32::consts::PI,
    ffi::c_void,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use windows::Win32::{
    Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM},
    Graphics::Gdi::{InvalidateRect, RDW_INVALIDATE, RDW_UPDATENOW, RedrawWindow, ValidateRect},
    Media::Audio::{
        AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK,
        IAudioCaptureClient, IAudioClient, IMMDevice, IMMDeviceEnumerator, MMDeviceEnumerator,
        WAVEFORMATEXTENSIBLE, eConsole, eRender,
    },
    Media::{
        KernelStreaming::WAVE_FORMAT_EXTENSIBLE,
        Multimedia::{KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT},
    },
    System::{Com::*, Threading::Sleep},
    UI::{Input::KeyboardAndMouse::*, WindowsAndMessaging::*},
};

mod renderer;
mod window;

use renderer::D2DRenderer;
use window::Window;

pub type ResultAny<T = ()> = color_eyre::Result<T>;

const FFT_SIZE: usize = 8192;
const HOP_SIZE: usize = FFT_SIZE / 4;
const TIME_FRAMES: usize = 256;
const MAX_HZ: f32 = 5000.0;
const UI_FPS: u32 = 360;
const WM_FFT_READY: u32 = WM_APP + 1;

/// Number of FFT bins to display, clamped to MAX_HZ.
fn display_bins(sample_rate: u32) -> usize {
    let bins = (MAX_HZ * FFT_SIZE as f32 / sample_rate as f32).ceil() as usize;
    bins.min(FFT_SIZE / 2)
}

fn cola_normalization(window: &[f32], hop_size: usize) -> Vec<f32> {
    let n = window.len();
    let mut norm = vec![0.0f32; n];
    let num_overlaps = (n + hop_size - 1) / hop_size;
    for k in 0..num_overlaps {
        let offset = k * hop_size;
        for i in 0..n {
            let j = (i + offset) % n;
            norm[i] += window[j] * window[j];
        }
    }

    norm
}

/// For windows that satisfy COLA (like Hann with hop = N/4),
/// the normalization is a constant. This returns that scalar.
fn cola_normalization_scalar(window: &[f32], hop_size: usize) -> f32 {
    let norm = cola_normalization(window, hop_size);
    norm.iter().sum::<f32>() / norm.len() as f32
}

fn sample_to_color(magnitude: f32) -> [u8; 4] {
    let magnitude = magnitude / FFT_SIZE as f32;
    let log_val = 20.0 * (magnitude + 1e-10).log10();

    let vmin = -100.0;
    let vmax = -30.0;
    let x = ((log_val - vmin) / (vmax - vmin)).clamp(0.0, 1.0);
    let x = x * x;

    let r = 1.0 * x + 2.0 * x * x - 2.0 * x * x * x;
    let g = 0.2 * x + 0.1 * x * x + 0.7 * x * x * x;
    let b = 3.8 * x - 8.9 * x * x + 5.8 * x * x * x;

    let r = (r.clamp(0.0, 1.0) * 255.0) as u8;
    let g = (g.clamp(0.0, 1.0) * 255.0) as u8;
    let b = (b.clamp(0.0, 1.0) * 255.0) as u8;

    [b, g, r, 255]
}

fn stft(buffer: &[f32], window: &[f32], fft: &Arc<dyn Fft<f32>>) -> Vec<Complex<f32>> {
    assert_eq!(buffer.len(), window.len());
    let mut windowed: Vec<Complex<f32>> = buffer
        .iter()
        .zip(window.iter())
        .map(|(&b, &w)| Complex::new(b * w, 0.0))
        .collect();

    fft.process(&mut windowed);

    return windowed;
}

fn hz_to_note(freq: f32) -> String {
    let n = (12.0 * (freq.max(1.0) / 440.0).log2() + 69.0).max(0.0);
    let n_int = n.round() as i32;

    let note_name = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ][(n_int % 12) as usize];
    let octave = n_int / 12 - 1;

    format!("{note_name}{octave}")
}

/// Returns: new magnitude column
fn process(buffer: &[f32], window: &[f32], fft: &Arc<dyn Fft<f32>>) -> Vec<f32> {
    assert_eq!(buffer.len(), FFT_SIZE);
    let windowed = stft(buffer, window, fft);
    windowed
        .iter()
        .take(FFT_SIZE / 2) // Use only positive frequencies (DC to Nyquist)
        .map(|x| x.norm())
        .collect()
}

unsafe extern "system" fn window_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_ERASEBKGND => return LRESULT(1),
        WM_FFT_READY => {
            // The audio thread posts this only after publishing a complete
            // FFT column. This keeps spectrum repaints synchronized to the
            // sample-rate / hop-size cadence instead of polling for them.
            let _ = unsafe { InvalidateRect(Some(hwnd), None, false) };
            LRESULT(0)
        }
        WM_PAINT => {
            let ws_ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut WindowState;
            if !ws_ptr.is_null() {
                let ws = unsafe { &mut *ws_ptr };

                // Snapshot shared state under the lock, then drop it before rendering.
                let snapshot = {
                    let guard = ws.state.lock().unwrap();
                    let mut window_rect = RECT::default();
                    unsafe { GetWindowRect(hwnd, &mut window_rect).ok() };
                    let width = window_rect.right - window_rect.left;
                    let height = window_rect.bottom - window_rect.top;

                    let mut screen_cursor = POINT::default();
                    unsafe { GetPhysicalCursorPos(&mut screen_cursor).ok() };
                    let mut window_rect = RECT::default();
                    unsafe { GetWindowRect(hwnd, &mut window_rect).ok() };
                    let mut cursor = POINT::default();
                    cursor.x = screen_cursor.x - window_rect.left;
                    cursor.y = screen_cursor.y - window_rect.top;

                    let mouse_in_client =
                        cursor.x >= 0 && cursor.x < width && cursor.y >= 0 && cursor.y < height;

                    let display_bins = display_bins(guard.sample_rate);
                    let ring_head = guard.ring_head;

                    let text = if mouse_in_client {
                        let bitmap_x = ((guard.mouse_x as f32 / width as f32) * display_bins as f32)
                            .floor() as usize;

                        let visual_y = ((guard.mouse_y as f32 / height as f32) * TIME_FRAMES as f32)
                            .floor() as usize;
                        let buffer_y = (ring_head + visual_y) % TIME_FRAMES;
                        let mag = if bitmap_x < display_bins && visual_y < TIME_FRAMES {
                            guard.magnitudes[buffer_y * (FFT_SIZE / 2) + bitmap_x]
                        } else {
                            0.0
                        };
                        let db_val = 20.0 * (mag * 2.0 / FFT_SIZE as f32 + 1e-10).log10();
                        if guard.is_paused {
                            format!(
                                "{:.0} Hz • {}\n{:.1} dB, {:.1e}",
                                guard.current_freq,
                                hz_to_note(guard.current_freq),
                                db_val,
                                mag
                            )
                        } else {
                            format!(
                                "{:.0} Hz • {}",
                                guard.current_freq,
                                hz_to_note(guard.current_freq)
                            )
                        }
                    } else {
                        String::new()
                    };

                    // Copy image data only when a new FFT column exists. UI-only
                    // repaints reuse the already uploaded bitmap.
                    if guard.image_generation != ws.last_image_generation {
                        ws.renderer.stage_image(&guard.image_data, ring_head);
                        ws.last_image_generation = guard.image_generation;
                    }

                    (
                        display_bins,
                        mouse_in_client,
                        text,
                        guard.mouse_x,
                        guard.mouse_y,
                    )
                }; // mutex released here

                let (db, mouse_in_client, text, mx, my) = snapshot;
                if let Err(error) =
                    ws.renderer
                        .paint(db, mouse_in_client, &text, mx as f32, my as f32)
                {
                    eprintln!("Render error: {error:?}");
                }

                let _ = unsafe { ValidateRect(Some(hwnd), None) };
            }

            LRESULT(0)
        }
        WM_MOUSEMOVE => {
            let ws_ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut WindowState;
            if !ws_ptr.is_null() {
                let ws = unsafe { &mut *ws_ptr };
                if let Ok(mut guard) = ws.state.lock() {
                    let mut screen_cursor = POINT::default();
                    unsafe { GetPhysicalCursorPos(&mut screen_cursor).ok() };
                    let mut window_rect = RECT::default();
                    unsafe { GetWindowRect(hwnd, &mut window_rect).ok() };
                    let mut cursor = POINT::default();
                    cursor.x = screen_cursor.x - window_rect.left;
                    cursor.y = screen_cursor.y - window_rect.top;
                    guard.mouse_x = cursor.x;
                    guard.mouse_y = cursor.y;

                    let width = window_rect.right - window_rect.left;
                    let height = window_rect.bottom - window_rect.top;
                    if width > 0 {
                        let freq = (guard.mouse_x as f32 / width as f32) * MAX_HZ;
                        guard.current_freq = freq;
                    }
                    if guard.dragging {
                        let delta_x = screen_cursor.x - guard.drag_start.x;
                        let delta_y = screen_cursor.y - guard.drag_start.y;
                        guard.drag_start = screen_cursor;

                        // Move the window
                        unsafe {
                            SetWindowPos(
                                hwnd,
                                None,
                                window_rect.left + delta_x,
                                window_rect.top + delta_y,
                                0,
                                0,
                                SWP_NOSIZE | SWP_NOZORDER,
                            )
                            .ok();
                        }
                    }

                    let do_resize = if guard.resizing {
                        let delta_x = screen_cursor.x - guard.resize_start.x;
                        let delta_y = screen_cursor.y - guard.resize_start.y;
                        guard.resize_start = screen_cursor;

                        let new_width = (width + delta_x).max(200);
                        let new_height = (height + delta_y).max(200);
                        Some((new_width, new_height))
                    } else {
                        None
                    };

                    // Drop the lock before resize (SetWindowPos sends WM_SIZE synchronously)
                    drop(guard);

                    if let Some((new_width, new_height)) = do_resize {
                        if let Err(error) = ws.renderer.resize(new_width as u32, new_height as u32)
                        {
                            eprintln!("Resize error: {error:?}");
                        }
                        unsafe {
                            SetWindowPos(
                                hwnd,
                                None,
                                0,
                                0,
                                new_width,
                                new_height,
                                SWP_NOMOVE | SWP_NOZORDER,
                            )
                            .ok();
                            _ = RedrawWindow(
                                Some(hwnd),
                                None,
                                None,
                                RDW_INVALIDATE | RDW_UPDATENOW,
                            );
                        }
                    }
                }
            }
            LRESULT(0)
        }
        WM_LBUTTONDOWN => {
            let ws_ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut WindowState;
            if !ws_ptr.is_null() {
                let ws = unsafe { &*ws_ptr };
                if let Ok(mut guard) = ws.state.lock() {
                    guard.dragging = true;
                    unsafe { GetPhysicalCursorPos(&mut guard.drag_start).ok() };
                    unsafe { SetCapture(hwnd) };
                }
            }
            LRESULT(0)
        }
        WM_LBUTTONUP => {
            let ws_ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut WindowState;
            if !ws_ptr.is_null() {
                let ws = unsafe { &*ws_ptr };
                if let Ok(mut guard) = ws.state.lock() {
                    guard.dragging = false;
                    unsafe { ReleaseCapture().unwrap() };
                }
            }
            LRESULT(0)
        }
        WM_RBUTTONDOWN => {
            let ws_ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut WindowState;
            if !ws_ptr.is_null() {
                let ws = unsafe { &*ws_ptr };
                if let Ok(mut guard) = ws.state.lock() {
                    guard.resizing = true;
                    unsafe { GetPhysicalCursorPos(&mut guard.resize_start).ok() };
                    unsafe { SetCapture(hwnd) };
                }
            }
            LRESULT(0)
        }
        WM_RBUTTONUP => {
            let ws_ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut WindowState;
            if !ws_ptr.is_null() {
                let ws = unsafe { &*ws_ptr };
                if let Ok(mut guard) = ws.state.lock() {
                    guard.resizing = false;
                    unsafe { ReleaseCapture().unwrap() };
                }
            }
            LRESULT(0)
        }
        WM_KEYDOWN => {
            match VIRTUAL_KEY(wparam.0 as u16) {
                VK_ESCAPE => {
                    unsafe { PostQuitMessage(0) };
                }
                VK_SPACE => {
                    let ws_ptr =
                        unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut WindowState;
                    if !ws_ptr.is_null() {
                        let ws = unsafe { &*ws_ptr };
                        if let Ok(mut guard) = ws.state.lock() {
                            guard.is_paused = !guard.is_paused;
                        }
                    }
                }
                VK_T => {
                    let ws_ptr =
                        unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut WindowState;
                    if !ws_ptr.is_null() {
                        let ws = unsafe { &*ws_ptr };
                        if let Ok(mut guard) = ws.state.lock() {
                            guard.is_always_on_top = !guard.is_always_on_top;
                            let hwnd_insert_after = if guard.is_always_on_top {
                                HWND_TOPMOST
                            } else {
                                HWND_NOTOPMOST
                            };
                            unsafe {
                                SetWindowPos(
                                    hwnd,
                                    Some(hwnd_insert_after),
                                    0,
                                    0,
                                    0,
                                    0,
                                    SWP_NOMOVE | SWP_NOSIZE,
                                )
                                .ok();
                            };
                        }
                    }
                }
                _ => {}
            }
            LRESULT(0)
        }
        WM_SIZE => {
            let width = (lparam.0 & 0xFFFF) as u32;
            let height = ((lparam.0 >> 16) & 0xFFFF) as u32;
            if width > 0 && height > 0 {
                let ws_ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut WindowState;
                if !ws_ptr.is_null() {
                    let ws = unsafe { &mut *ws_ptr };
                    if let Err(error) = ws.renderer.resize(width, height) {
                        eprintln!("Resize error: {error:?}");
                    }
                }
            }
            LRESULT(0)
        }
        WM_DESTROY => {
            unsafe { PostQuitMessage(0) };
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

fn audio_thread_loop(state: &Arc<Mutex<AppState>>) -> ResultAny {
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;

        // A WASAPI client can become invalid when the default device changes or
        // the audio service restarts. Recreate the whole client instead of
        // allowing one transient error to permanently kill the worker thread.
        loop {
            if let Err(error) = audio_capture_session(state) {
                eprintln!("Audio session stopped: {error:?}; retrying");
                Sleep(250);
            }
        }
    }
}

fn audio_capture_session(state: &Arc<Mutex<AppState>>) -> ResultAny {
    unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
        let device: IMMDevice = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
        let audio_client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;

        let format = audio_client.GetMixFormat()?;
        let channels = (*format).nChannels as usize;
        let sample_rate = (*format).nSamplesPerSec;
        let is_float32 = if (*format).wFormatTag == WAVE_FORMAT_IEEE_FLOAT as u16 {
            (*format).wBitsPerSample == 32
                && (*format).nBlockAlign == (channels * std::mem::size_of::<f32>()) as u16
        } else if (*format).wFormatTag == WAVE_FORMAT_EXTENSIBLE as u16 && (*format).cbSize >= 22 {
            let extensible = format as *const WAVEFORMATEXTENSIBLE;
            std::ptr::read_unaligned(std::ptr::addr_of!((*extensible).SubFormat))
                == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
                && (*format).wBitsPerSample == 32
                && (*format).nBlockAlign == (channels * std::mem::size_of::<f32>()) as u16
        } else {
            false
        };
        if channels == 0 || !is_float32 {
            CoTaskMemFree(Some(format as *const c_void));
            return Err(eyre!("audio device format is not interleaved 32-bit float"));
        }

        // GetMixFormat returns an allocated WAVEFORMATEX. Initialize copies it,
        // so release it immediately after Initialize, including on failure.
        let initialize_result = audio_client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            0,
            0,
            format,
            None,
        );
        CoTaskMemFree(Some(format as *const c_void));
        initialize_result?;

        state.lock().unwrap().sample_rate = sample_rate;

        let mut buffer = vec![0.0f32; FFT_SIZE];
        let mut buffer_head = 0;
        let mut hann_window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / FFT_SIZE as f32).cos()))
            .collect();
        let window_norm = cola_normalization_scalar(&hann_window, HOP_SIZE).sqrt();
        hann_window.iter_mut().for_each(|w| *w /= window_norm);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(buffer.len());

        let capture: IAudioCaptureClient = audio_client.GetService()?;
        audio_client.Start()?;

        loop {
            let packet_frames = capture.GetNextPacketSize()?;
            if packet_frames == 0 {
                Sleep(1);
                continue;
            }

            let mut data_ptr = std::ptr::null_mut();
            let mut frames = 0;
            let mut flags = 0;
            capture.GetBuffer(&mut data_ptr, &mut frames, &mut flags, None, None)?;

            // Copy the packet and release WASAPI's buffer before doing any FFT
            // or mutex work. This prevents capture-buffer overruns.
            let samples: Vec<f32> = if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
                vec![0.0; frames as usize * channels]
            } else {
                if data_ptr.is_null() {
                    capture.ReleaseBuffer(frames)?;
                    return Err(eyre!("WASAPI returned a null audio buffer"));
                }
                let raw_samples =
                    std::slice::from_raw_parts(data_ptr as *const f32, frames as usize * channels);
                raw_samples
                    .chunks_exact(channels)
                    .map(|chunk| chunk.iter().copied().sum::<f32>() / channels as f32)
                    .collect()
            };
            capture.ReleaseBuffer(frames)?;

            let mut sample_idx = 0;
            while sample_idx < samples.len() {
                // Pausing freezes the displayed spectrum and discards incoming
                // samples. Resetting the partial frame also prevents the
                // full-buffer/zero-space infinite loop.
                if state.lock().unwrap().is_paused {
                    buffer_head = 0;
                    break;
                }

                if buffer_head == buffer.len() {
                    let mag_column = process(&buffer, &hann_window, &fft);
                    let rgba_column: Vec<u8> = mag_column
                        .iter()
                        .flat_map(|&mag| sample_to_color(mag))
                        .collect();

                    buffer.copy_within(HOP_SIZE.., 0);
                    buffer_head = FFT_SIZE - HOP_SIZE;

                    const W: usize = FFT_SIZE / 2;
                    let mut guard = state.lock().unwrap();
                    guard.ring_head = (guard.ring_head + TIME_FRAMES - 1) % TIME_FRAMES;
                    let row = guard.ring_head;
                    guard.image_data[row * W * 4..(row + 1) * W * 4].copy_from_slice(&rgba_column);
                    guard.magnitudes[row * W..(row + 1) * W].copy_from_slice(&mag_column);
                    guard.image_generation = guard.image_generation.wrapping_add(1);
                    let hwnd = HWND(guard.hwnd as *mut c_void);
                    drop(guard);
                    let _ = PostMessageW(Some(hwnd), WM_FFT_READY, WPARAM(0), LPARAM(0));
                }

                let space = buffer.len() - buffer_head;
                let n = (samples.len() - sample_idx).min(space);
                buffer[buffer_head..buffer_head + n]
                    .copy_from_slice(&samples[sample_idx..sample_idx + n]);
                buffer_head += n;
                sample_idx += n;
            }
        }
    }
}

struct AppState {
    hwnd: isize,
    image_data: Vec<u8>,
    magnitudes: Vec<f32>,
    image_generation: u64,
    ring_head: usize,
    sample_rate: u32,
    dragging: bool,
    drag_start: POINT,
    resizing: bool,
    resize_start: POINT,
    is_always_on_top: bool,
    mouse_x: i32,
    mouse_y: i32,
    current_freq: f32,
    is_paused: bool,
}

/// Stored in GWLP_USERDATA. Renderer is only accessed from the UI thread
/// so it doesn't need the mutex. Shared audio state is behind the Mutex.
struct WindowState {
    state: Arc<Mutex<AppState>>,
    renderer: D2DRenderer,
    last_image_generation: u64,
}

fn run_app() -> ResultAny {
    unsafe {
        let window = Window::new(1024, 768, Some(window_proc))?;

        let d2d_renderer = D2DRenderer::new(window.hwnd(), FFT_SIZE, TIME_FRAMES)?;

        let state = Arc::new(Mutex::new(AppState {
            hwnd: window.hwnd().0 as isize,
            image_data: vec![0u8; (FFT_SIZE / 2) * TIME_FRAMES * 4],
            magnitudes: vec![0.0; (FFT_SIZE / 2) * TIME_FRAMES],
            image_generation: 0,
            ring_head: 0,
            sample_rate: 48000,
            dragging: false,
            drag_start: POINT::default(),
            resizing: false,
            resize_start: POINT::default(),
            is_always_on_top: false,
            mouse_x: 0,
            mouse_y: 0,
            current_freq: 0.0,
            is_paused: false,
        }));

        let ws = Box::new(WindowState {
            state: state.clone(),
            renderer: d2d_renderer,
            last_image_generation: u64::MAX,
        });
        window.set_user_data(Box::into_raw(ws) as isize);

        std::thread::spawn(move || {
            if let Err(e) = audio_thread_loop(&state) {
                eprintln!("Audio Error: {:?}", e);
            }
        });

        let frame_period = Duration::from_nanos(1_000_000_000 / UI_FPS as u64);
        let mut next_frame = Instant::now();

        while window.handle_events() {
            let now = Instant::now();
            if now >= next_frame {
                let _ = InvalidateRect(Some(window.hwnd()), None, false);

                // Keep UI-only updates responsive at the target display rate.
                // The FFT image itself is staged only when image_generation
                // changes, so this does not repeatedly copy the 4 MiB spectrum
                // buffer.
                next_frame += frame_period;
                if next_frame <= now {
                    // Do not accumulate lateness after a blocked or slow frame.
                    next_frame = now + frame_period;
                }
            } else {
                // Wake immediately for input, but otherwise wait until the next
                // frame deadline instead of polling at maximum speed.
                window.wait_for_events(next_frame - now);
            }
        }
    }

    Ok(())
}

use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
};

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    unsafe { SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2).ok() };
    Ok(run_app()?)
}
