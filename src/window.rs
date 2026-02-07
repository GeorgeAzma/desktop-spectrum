use crate::ResultAny;
use windows::{
    Win32::{
        Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM},
        Graphics::Gdi::*,
        System::LibraryLoader::*,
        UI::WindowsAndMessaging::*,
    },
    core::*,
};

pub struct Window {
    hwnd: HWND,
}

impl Window {
    pub fn new(
        width: i32,
        height: i32,
        window_proc: Option<unsafe extern "system" fn(HWND, u32, WPARAM, LPARAM) -> LRESULT>,
    ) -> ResultAny<Self> {
        let hinstance = unsafe { GetModuleHandleW(None) }?;
        let class_name = w!("SpectrogramClass");
        let wc = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: window_proc,
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: HINSTANCE(hinstance.0),
            hIcon: unsafe { LoadIconW(None, IDI_APPLICATION) }?,
            hCursor: unsafe { LoadCursorW(None, IDC_ARROW) }?,
            hbrBackground: HBRUSH(std::ptr::null_mut()),
            lpszMenuName: PCWSTR::null(),
            lpszClassName: class_name,
        };
        unsafe { RegisterClassW(&wc) };

        let hwnd = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE(0),
                class_name,
                w!("Spectrogram"),
                WS_POPUP | WS_VISIBLE,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                width,
                height,
                None,
                None,
                Some(HINSTANCE(GetModuleHandleW(None)?.0)),
                None,
            )
        }?;

        _ = unsafe { UpdateWindow(hwnd) };

        Ok(Self { hwnd })
    }

    pub fn set_user_data(&self, user_data: isize) {
        if user_data != 0 {
            unsafe { SetWindowLongPtrW(self.hwnd, GWLP_USERDATA, user_data) };
        }
    }

    pub fn width(&self) -> i32 {
        let mut rect = RECT::default();
        unsafe { GetWindowRect(self.hwnd, &mut rect) }.unwrap();
        rect.right - rect.left
    }

    pub fn height(&self) -> i32 {
        let mut rect = RECT::default();
        unsafe { GetWindowRect(self.hwnd, &mut rect) }.unwrap();
        rect.bottom - rect.top
    }

    pub fn set_width(&self, width: i32) {
        unsafe {
            SetWindowPos(
                self.hwnd,
                None,
                0,
                0,
                width,
                self.height(),
                SWP_NOMOVE | SWP_NOZORDER,
            )
        }
        .unwrap();
    }

    pub fn set_height(&self, height: i32) {
        unsafe {
            SetWindowPos(
                self.hwnd,
                None,
                0,
                0,
                self.width(),
                height,
                SWP_NOMOVE | SWP_NOZORDER,
            )
        }
        .unwrap();
    }

    pub fn handle_events(&self) -> bool {
        let mut msg = MSG::default();
        // Block until a message arrives, avoiding busy-spin
        unsafe { WaitMessage().ok() };
        while unsafe { PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() } {
            if msg.message == WM_QUIT {
                return false;
            }
            unsafe {
                _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        true
    }

    pub fn hwnd(&self) -> HWND {
        self.hwnd
    }
}
