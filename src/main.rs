// #![windows_subsystem = "windows"]
#![allow(dead_code)]
pub use color_eyre::eyre::eyre;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::{
    f32::consts::PI,
    ffi::c_void,
    ptr::null_mut,
    sync::{Arc, Mutex},
};
use windows::{
    Win32::{
        Foundation::{COLORREF, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM},
        Graphics::Gdi::*,
        Media::Audio::{
            AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK, IAudioCaptureClient,
            IAudioClient, IMMDevice, IMMDeviceEnumerator, MMDeviceEnumerator, eConsole, eRender,
        },
        System::{Com::*, Threading::Sleep},
        UI::{Input::KeyboardAndMouse::*, WindowsAndMessaging::*},
    },
    core::*,
};

// mod renderer;
mod window;

// use renderer::Renderer;
use window::Window;

pub type ResultAny<T = ()> = color_eyre::Result<T>;

const fn rgb(r: u8, g: u8, b: u8) -> COLORREF {
    COLORREF(r as u32 | (g as u32) << 8 | (b as u32) << 16)
}

const FFT_SIZE: usize = 2048;
const HOP_SIZE: usize = FFT_SIZE / 4;
const TIME_FRAMES: usize = 768;

fn sample_to_color(magnitude: f32) -> [u8; 3] {
    let magnitude = magnitude * 2.0 / FFT_SIZE as f32;
    let log_val = 20.0 * (magnitude + 1e-10).log10();

    let vmin = -90.0;
    let vmax = -20.0;
    let x = ((log_val - vmin) / (vmax - vmin)).clamp(0.0, 1.0);

    let r = 1.0 * x + 2.0 * x * x - 2.0 * x * x * x;
    let g = 0.2 * x + 0.1 * x * x + 0.7 * x * x * x;
    let b = 3.8 * x - 8.9 * x * x + 5.8 * x * x * x;

    let r = (r.clamp(0.0, 1.0) * 255.0) as u8;
    let g = (g.clamp(0.0, 1.0) * 255.0) as u8;
    let b = (b.clamp(0.0, 1.0) * 255.0) as u8;

    [b, g, r]
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
        WM_PAINT => {
            let state_ptr =
                unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const Mutex<AppState>;
            if !state_ptr.is_null() {
                let state = unsafe { &*state_ptr };
                if let Ok(mut guard) = state.lock() {
                    // guard.renderer.lock().unwrap().render().unwrap();

                    let mut ps = PAINTSTRUCT::default();
                    let hdc = unsafe { BeginPaint(hwnd, &mut ps) };

                    let mut client_rect = RECT::default();
                    unsafe { GetClientRect(hwnd, &mut client_rect).ok() };
                    let width = client_rect.right - client_rect.left;
                    let height = client_rect.bottom - client_rect.top;

                    let mut cursor_pos = POINT::default();
                    unsafe { GetCursorPos(&mut cursor_pos).ok() };
                    let mut client_cursor = cursor_pos;
                    _ = unsafe { ScreenToClient(hwnd, &mut client_cursor).ok() };
                    let mouse_in_client = client_cursor.x >= 0
                        && client_cursor.x < width
                        && client_cursor.y >= 0
                        && client_cursor.y < height;

                    if mouse_in_client {
                        guard.mouse_x = client_cursor.x;
                        guard.mouse_y = client_cursor.y;
                        let freq = (client_cursor.x as f32 / width as f32)
                            * (guard.sample_rate as f32 / 2.0);
                        guard.current_freq = freq;
                    }

                    let bmi = BITMAPINFO {
                        bmiHeader: BITMAPINFOHEADER {
                            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                            biWidth: FFT_SIZE as i32 / 2,
                            biHeight: -(TIME_FRAMES as i32),
                            biPlanes: 1,
                            biBitCount: 24,
                            biCompression: BI_RGB.0,
                            ..Default::default()
                        },
                        ..Default::default()
                    };

                    unsafe {
                        SetStretchBltMode(hdc, HALFTONE);
                        _ = SetBrushOrgEx(hdc, 0, 0, None);
                        StretchDIBits(
                            hdc,
                            0,
                            0,
                            width,
                            height,
                            0,
                            0,
                            FFT_SIZE as i32 / 2,
                            TIME_FRAMES as i32,
                            Some(guard.image_data.as_ptr() as *const _),
                            &bmi,
                            DIB_RGB_COLORS,
                            SRCCOPY,
                        )
                    };

                    if mouse_in_client {
                        unsafe { SetBkMode(hdc, TRANSPARENT) };

                        let hfont = unsafe {
                            CreateFontW(
                                20,
                                0,
                                0,
                                0,
                                FW_SEMIBOLD.0 as i32,
                                0,
                                0,
                                0,
                                DEFAULT_CHARSET,
                                OUT_DEFAULT_PRECIS,
                                CLIP_DEFAULT_PRECIS,
                                DEFAULT_QUALITY,
                                DEFAULT_PITCH.0 as u32 | FF_DONTCARE.0 as u32,
                                w!("Segoe UI"),
                            )
                        };
                        unsafe { SelectObject(hdc, HGDIOBJ(hfont.0)) };

                        let bitmap_x = ((guard.mouse_x as f32 / width as f32)
                            * (FFT_SIZE as f32 / 2.0))
                            .floor() as usize;
                        let bitmap_y = ((guard.mouse_y as f32 / height as f32) * TIME_FRAMES as f32)
                            .floor() as usize;
                        let mag = if bitmap_x < FFT_SIZE / 2 && bitmap_y < TIME_FRAMES {
                            guard.magnitudes[bitmap_y * (FFT_SIZE / 2) + bitmap_x]
                        } else {
                            0.0
                        };
                        let db = 20.0 * (mag * 2.0 / FFT_SIZE as f32 + 1e-10).log10();

                        let text = if guard.is_paused {
                            format!("{:.0} Hz, {:.1} dB, {:.1e}", guard.current_freq, db, mag)
                        } else {
                            format!("{:.0} Hz", guard.current_freq)
                        };
                        let wide_text: Vec<u16> =
                            text.encode_utf16().chain(std::iter::once(0)).collect();
                        unsafe { SetTextColor(hdc, rgb(0, 0, 0)) };
                        _ = unsafe {
                            TextOutW(hdc, guard.mouse_x + 2, guard.mouse_y - 18, &wide_text)
                        };
                        unsafe { SetTextColor(hdc, rgb(255, 255, 255)) };
                        _ = unsafe { TextOutW(hdc, guard.mouse_x, guard.mouse_y - 20, &wide_text) };

                        _ = unsafe { DeleteObject(HGDIOBJ(hfont.0)) };
                    }
                    _ = unsafe { EndPaint(hwnd, &ps) };
                }
            }

            LRESULT(0)
        }
        WM_MOUSEMOVE => {
            let state_ptr =
                unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const Mutex<AppState>;
            if !state_ptr.is_null() {
                let state = unsafe { &*state_ptr };
                if let Ok(mut guard) = state.lock() {
                    let x = (lparam.0 & 0xFFFF) as i32;
                    let y = ((lparam.0 >> 16) & 0xFFFF) as i32;
                    guard.mouse_x = x;
                    guard.mouse_y = y;

                    let mut client_rect = RECT::default();
                    unsafe { GetClientRect(hwnd, &mut client_rect).ok() };
                    let client_width = client_rect.right - client_rect.left;
                    if client_width > 0 {
                        let freq =
                            (x as f32 / client_width as f32) * (guard.sample_rate as f32 / 2.0);
                        guard.current_freq = freq;
                    }
                    if guard.dragging {
                        let mut current_pos = POINT::default();
                        unsafe { GetCursorPos(&mut current_pos).ok() };
                        let delta_x = current_pos.x - guard.drag_start.x;
                        let delta_y = current_pos.y - guard.drag_start.y;
                        guard.drag_start = current_pos;

                        // Move the window
                        let mut rect = RECT::default();
                        unsafe { GetWindowRect(hwnd, &mut rect).ok() };
                        unsafe {
                            SetWindowPos(
                                hwnd,
                                None,
                                rect.left + delta_x,
                                rect.top + delta_y,
                                0,
                                0,
                                SWP_NOSIZE | SWP_NOZORDER,
                            )
                            .ok();
                        }
                    }

                    if guard.resizing {
                        let mut current_pos = POINT::default();
                        unsafe { GetCursorPos(&mut current_pos).ok() };
                        let delta_x = current_pos.x - guard.resize_start.x;
                        let delta_y = current_pos.y - guard.resize_start.y;
                        guard.resize_start = current_pos;

                        // Resize the window
                        let mut rect = RECT::default();
                        unsafe { GetWindowRect(hwnd, &mut rect).ok() };
                        let new_width = ((rect.right - rect.left) + delta_x).max(200);
                        let new_height = ((rect.bottom - rect.top) + delta_y).max(200);
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
                        }
                    }
                }
            }
            LRESULT(0)
        }
        WM_LBUTTONDOWN => {
            let state_ptr =
                unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const Mutex<AppState>;
            if !state_ptr.is_null() {
                let state = unsafe { &*state_ptr };
                if let Ok(mut guard) = state.lock() {
                    guard.dragging = true;
                    unsafe { GetCursorPos(&mut guard.drag_start).ok() };
                    unsafe { SetCapture(hwnd) };
                }
            }
            LRESULT(0)
        }
        WM_LBUTTONUP => {
            let state_ptr =
                unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const Mutex<AppState>;
            if !state_ptr.is_null() {
                let state = unsafe { &*state_ptr };
                if let Ok(mut guard) = state.lock() {
                    guard.dragging = false;
                    unsafe { ReleaseCapture().unwrap() };
                }
            }
            LRESULT(0)
        }
        WM_RBUTTONDOWN => {
            let state_ptr =
                unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const Mutex<AppState>;
            if !state_ptr.is_null() {
                let state = unsafe { &*state_ptr };
                if let Ok(mut guard) = state.lock() {
                    guard.resizing = true;
                    unsafe { GetCursorPos(&mut guard.resize_start).ok() };
                    unsafe { SetCapture(hwnd) };
                }
            }
            LRESULT(0)
        }
        WM_RBUTTONUP => {
            let state_ptr =
                unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const Mutex<AppState>;
            if !state_ptr.is_null() {
                let state = unsafe { &*state_ptr };
                if let Ok(mut guard) = state.lock() {
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
                    let state_ptr =
                        unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const Mutex<AppState>;
                    if !state_ptr.is_null() {
                        let state = unsafe { &*state_ptr };
                        if let Ok(mut guard) = state.lock() {
                            guard.is_paused = !guard.is_paused;
                        }
                    }
                    _ = unsafe { InvalidateRect(Some(hwnd), None, false) };
                }
                VK_T => {
                    let state_ptr =
                        unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const Mutex<AppState>;
                    if !state_ptr.is_null() {
                        let state = unsafe { &*state_ptr };
                        if let Ok(mut guard) = state.lock() {
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
        WM_DESTROY => {
            unsafe { PostQuitMessage(0) };
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

fn audio_thread_loop(hwnd: usize, state: &Arc<Mutex<AppState>>) -> ResultAny {
    let hwnd = HWND(hwnd as *mut c_void);
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;

        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
        let device: IMMDevice = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;

        let audio_client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;

        let format = audio_client.GetMixFormat()?;
        let channels = (*format).nChannels;
        let sample_rate = (*format).nSamplesPerSec;
        state.lock().unwrap().sample_rate = sample_rate;

        let mut buffer = vec![0f32; FFT_SIZE];
        let mut buffer_head = 0;
        let hann_window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / FFT_SIZE as f32).cos()))
            .collect();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(buffer.len());

        audio_client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            0,
            0,
            format,
            None,
        )?;

        let capture: IAudioCaptureClient = audio_client.GetService()?;
        audio_client.Start()?;

        loop {
            let packet_frames = capture.GetNextPacketSize()?;

            if packet_frames == 0 {
                Sleep(0);
                continue;
            }

            let mut data_ptr = null_mut();
            let mut frames = 0;
            let mut flags = 0;

            capture.GetBuffer(&mut data_ptr, &mut frames, &mut flags, None, None)?;

            let raw_samples = std::slice::from_raw_parts(
                data_ptr as *const f32,
                frames as usize * channels as usize,
            );
            let samples: Vec<f32> = raw_samples
                .chunks_exact(channels as usize)
                .map(|chunk| {
                    let sum: f32 = chunk.iter().map(|&s| s).sum();
                    sum / chunk.len() as f32
                })
                .collect();

            let mut sample_idx = 0;
            while sample_idx < samples.len() {
                let space = buffer.len() - buffer_head;
                let n = (samples.len() - sample_idx).min(space);
                buffer[buffer_head..buffer_head + n]
                    .copy_from_slice(&samples[sample_idx..sample_idx + n]);
                buffer_head += n;
                sample_idx += n;

                // If buffer is full, process STFT and shift for overlap
                if buffer_head >= buffer.len() && !state.lock().unwrap().is_paused {
                    let mag_column = process(&buffer, &hann_window, &fft);
                    let rgb_column: Vec<u8> = mag_column
                        .iter()
                        .flat_map(|&mag| sample_to_color(mag))
                        .collect();
                    // Shift buffer by HOP_SIZE for overlap
                    buffer.copy_within(HOP_SIZE.., 0);
                    buffer_head = FFT_SIZE - HOP_SIZE;

                    // Add new row to the image, scroll old rows down
                    const W: usize = FFT_SIZE / 2;
                    const H: usize = TIME_FRAMES;
                    {
                        let mut guard = state.lock().unwrap();
                        for i in (1..H).rev() {
                            let src_start = (i - 1) * W;
                            let dst_start = i * W;
                            guard
                                .image_data
                                .copy_within(src_start * 3..(src_start + W) * 3, dst_start * 3);
                            guard
                                .magnitudes
                                .copy_within(src_start..src_start + W, dst_start);
                        }
                        // Add new row at the top
                        guard.image_data[0..W * 3].copy_from_slice(&rgb_column);
                        guard.magnitudes[0..W].copy_from_slice(&mag_column);
                    }

                    InvalidateRect(Some(hwnd), None, false).unwrap();
                }
            }

            capture.ReleaseBuffer(frames)?;
        }
    }
}

struct AppState {
    image_data: Vec<u8>,
    magnitudes: Vec<f32>,
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
    // renderer: Arc<Mutex<Renderer>>,
}

fn run_app() -> ResultAny {
    unsafe {
        let window = Window::new(1024, 768, Some(window_proc))?;
        // let renderer = Renderer::new(window.hwnd(), window.width() as u32, window.height() as u32)?;

        let state = Arc::new(Mutex::new(AppState {
            image_data: vec![0u8; (FFT_SIZE / 2) * TIME_FRAMES * 3],
            magnitudes: vec![0.0; (FFT_SIZE / 2) * TIME_FRAMES],
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
            // renderer: Arc::new(Mutex::new(renderer)),
        }));

        window.set_user_data(Arc::into_raw(state.clone()) as isize);

        let sendable_hwnd = window.hwnd().0 as usize;
        std::thread::spawn(move || {
            if let Err(e) = audio_thread_loop(sendable_hwnd, &state) {
                eprintln!("Audio Error: {:?}", e);
            }
        });

        while window.handle_events() {}

        CoUninitialize();
    }

    Ok(())
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    _ = unsafe { SetProcessDPIAware() };
    Ok(run_app()?)
}
