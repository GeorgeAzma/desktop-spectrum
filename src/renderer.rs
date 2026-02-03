use std::{ffi::CStr, mem::ManuallyDrop, ptr::null_mut};

use crate::{ResultAny, eyre};
use windows::{
    Win32::{
        Foundation::{HANDLE, HWND, RECT},
        Graphics::{
            Direct3D::Fxc::D3DCompile, Direct3D::*, Direct3D12::*, Dxgi::Common::*, Dxgi::*,
        },
        System::Threading::{CreateEventW, INFINITE, WaitForSingleObject},
    },
    core::*,
};

fn create_device() -> ResultAny<ID3D12Device> {
    let mut device: Option<ID3D12Device> = None;

    unsafe {
        let mut debug: Option<ID3D12Debug> = None;
        D3D12GetDebugInterface(&mut debug)?;
        debug.unwrap().EnableDebugLayer();

        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)
            .expect("D3D12CreateDevice failed")
    };

    Ok(device.unwrap())
}

fn create_cmd_queue(device: &ID3D12Device) -> ResultAny<ID3D12CommandQueue> {
    let desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0 as i32,
        Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
        NodeMask: 0,
    };

    unsafe { Ok(device.CreateCommandQueue(&desc)?) }
}

fn create_swapchain(
    hwnd: HWND,
    queue: &ID3D12CommandQueue,
    width: u32,
    height: u32,
) -> ResultAny<IDXGISwapChain3> {
    let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1()? };
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        BufferCount: 2,
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let sc1 = unsafe { factory.CreateSwapChainForHwnd(queue, hwnd, &desc, None, None)? };
    Ok(sc1.cast()?)
}

fn create_descriptor_heap(
    device: &ID3D12Device,
    num_descriptors: u32,
) -> ResultAny<ID3D12DescriptorHeap> {
    unsafe {
        Ok(device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: num_descriptors,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        })?)
    }
}

fn err_blob_to_str<'a>(err_blob: ID3DBlob) -> &'a str {
    let err_ptr = unsafe { err_blob.GetBufferPointer() };
    let err_size = unsafe { err_blob.GetBufferSize() };
    let err_msg = unsafe {
        std::str::from_utf8(std::slice::from_raw_parts(err_ptr as *const u8, err_size))
            .unwrap_or("Failed to decode error message")
    };
    err_msg
}

fn create_root_signature(device: &ID3D12Device) -> ResultAny<ID3D12RootSignature> {
    let root_sig_desc = D3D12_ROOT_SIGNATURE_DESC {
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        ..Default::default()
    };
    let mut sig_blob = None;
    let mut err_blob = None;
    unsafe {
        D3D12SerializeRootSignature(
            &root_sig_desc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut sig_blob,
            Some(&mut err_blob),
        )
    }?;
    if let Some(err_blob) = err_blob {
        let err = err_blob_to_str(err_blob);
        eprintln!("Shader compilation error: {err}");
        Err(eyre!(err))
    } else {
        if let Some(sig_blob) = sig_blob.as_ref() {
            let err_ptr = unsafe { sig_blob.GetBufferPointer() };
            let err_size = unsafe { sig_blob.GetBufferSize() };
            let sig_slice = unsafe { std::slice::from_raw_parts(err_ptr as *const u8, err_size) };
            let root_sig: ID3D12RootSignature =
                unsafe { device.CreateRootSignature(0, sig_slice) }?;
            Ok(root_sig)
        } else {
            Err(eyre!("failed to compile"))
        }
    }
}

fn compile_shader(source: &str, entry: &CStr, target: &CStr) -> ResultAny<ID3DBlob> {
    use windows::core::PCSTR;

    let mut code_blob = None;
    let mut err_blob = None;

    unsafe {
        D3DCompile(
            source.as_ptr() as *const _,
            source.len(),
            None,
            None,
            None,
            PCSTR(entry.as_ptr() as *const u8),
            PCSTR(target.as_ptr() as *const u8),
            0, // D3DCOMPILE_DEBUG, D3DCOMPILE_OPTIMIZATION_LEVEL3
            0,
            &mut code_blob,
            Some(&mut err_blob),
        )?
    };

    if let Some(err_blob) = err_blob {
        let err_msg = err_blob_to_str(err_blob);
        eprintln!("Shader compilation error: {}", err_msg);
        Err(eyre!(err_msg))
    } else {
        Ok(code_blob.unwrap())
    }
}

pub struct Renderer {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    swapchain: IDXGISwapChain3,
    descriptor_heap: ID3D12DescriptorHeap,
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    cmd_allocator: ID3D12CommandAllocator,
    cmd_list: ID3D12GraphicsCommandList,
    vb: ID3D12Resource,
    vb_view: D3D12_VERTEX_BUFFER_VIEW,
    render_targets: [Option<ID3D12Resource>; 2],
    fence: ID3D12Fence,
    fence_event: HANDLE,
    fence_value: u64,
}

unsafe impl Send for Renderer {}
unsafe impl Sync for Renderer {}

impl Renderer {
    pub fn new(hwnd: HWND, width: u32, height: u32) -> ResultAny<Self> {
        let device = create_device()?;
        let queue = create_cmd_queue(&device)?;
        let swapchain = create_swapchain(hwnd, &queue, width, height)?;
        let descriptor_heap = create_descriptor_heap(&device, 2)?;

        let rtv_size =
            unsafe { device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) };
        let mut rtv_handle = unsafe { descriptor_heap.GetCPUDescriptorHandleForHeapStart() };
        let mut render_targets: [Option<ID3D12Resource>; 2] = [None, None];
        for i in 0..2 {
            render_targets[i] = unsafe { swapchain.GetBuffer(i as u32).ok() };
            unsafe {
                device.CreateRenderTargetView(render_targets[i].as_ref().unwrap(), None, rtv_handle)
            };
            rtv_handle.ptr += rtv_size as usize;
        }

        let root_sig = create_root_signature(&device)?;
        let shader_source = std::fs::read_to_string("src/shader.hlsl")?;
        let vs_blob = compile_shader(&shader_source, c"vs_main", c"vs_5_1")?;
        let ps_blob = compile_shader(&shader_source, c"ps_main", c"ps_5_1")?;

        let input_layout = [
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("POSITION"),
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                AlignedByteOffset: 0,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                ..Default::default()
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("COLOR"),
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                AlignedByteOffset: 12,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                ..Default::default()
            },
        ];

        let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: input_layout.as_ptr(),
                NumElements: input_layout.len() as u32,
            },
            pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: unsafe { vs_blob.GetBufferPointer() },
                BytecodeLength: unsafe { vs_blob.GetBufferSize() },
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: unsafe { ps_blob.GetBufferPointer() },
                BytecodeLength: unsafe { ps_blob.GetBufferSize() },
            },
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                ..Default::default()
            },
            BlendState: D3D12_BLEND_DESC {
                RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC {
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                }; 8],
                ..Default::default()
            },
            DepthStencilState: Default::default(),
            SampleMask: u32::MAX,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: [
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
            ],
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let pso: ID3D12PipelineState = unsafe { device.CreateGraphicsPipelineState(&pso_desc) }?;

        let cmd_allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }?;
        let cmd_list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &cmd_allocator, None)
        }?;
        unsafe { cmd_list.Close() }?; // start closed

        // ---------------- vertex buffer (UPLOAD heap) ----------------
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Vertex {
            pos: [f32; 3],
            col: [f32; 3],
        }
        let mut vb: Option<ID3D12Resource> = None;
        (unsafe {
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_UPLOAD,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Alignment: 0,
                    Width: size_of::<Vertex>() as u64 * 3,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_UNKNOWN,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_NONE,
                },
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut vb,
            )
        })?;
        let vb = vb.unwrap();
        let mut ptr = null_mut();
        (unsafe { vb.Map(0, None, Some(&mut ptr)) })?;
        const TRI: [Vertex; 3] = [
            Vertex {
                pos: [0.0, 0.5, 0.0],
                col: [1.0, 0.0, 0.0],
            },
            Vertex {
                pos: [0.5, -0.5, 0.0],
                col: [0.0, 1.0, 0.0],
            },
            Vertex {
                pos: [-0.5, -0.5, 0.0],
                col: [0.0, 0.0, 1.0],
            },
        ];
        unsafe { std::ptr::copy_nonoverlapping(TRI.as_ptr(), ptr as *mut Vertex, 3) };
        unsafe { vb.Unmap(0, None) };
        let vb_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: unsafe { vb.GetGPUVirtualAddress() },
            SizeInBytes: (size_of::<Vertex>() * 3) as u32,
            StrideInBytes: size_of::<Vertex>() as u32,
        };
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }?;
        let fence_event = unsafe { CreateEventW(None, false, false, None) }?;
        let fence_value = 1u64;

        Ok(Self {
            device,
            queue,
            swapchain,
            descriptor_heap,
            root_sig,
            pso,
            cmd_allocator,
            cmd_list,
            vb,
            vb_view,
            render_targets,
            fence,
            fence_event,
            fence_value,
        })
    }

    pub fn render(&mut self) -> ResultAny<()> {
        let cmd = &self.cmd_list;

        unsafe { self.cmd_allocator.Reset() }?;
        unsafe { cmd.Reset(&self.cmd_allocator, Some(&self.pso)) }?;

        let frame = unsafe { self.swapchain.GetCurrentBackBufferIndex() };
        let rt = self.render_targets[frame as usize].as_ref().unwrap();

        unsafe {
            cmd.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(rt.clone())),
                        StateBefore: D3D12_RESOURCE_STATE_PRESENT,
                        StateAfter: D3D12_RESOURCE_STATE_RENDER_TARGET,
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    }),
                },
            }])
        };

        let mut handle = unsafe { self.descriptor_heap.GetCPUDescriptorHandleForHeapStart() };
        let rtv_size = unsafe {
            self.device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
        };
        handle.ptr += frame as usize * rtv_size as usize;
        unsafe { cmd.OMSetRenderTargets(1, Some(&handle), false, None) };
        unsafe { cmd.ClearRenderTargetView(handle, &[0.1, 0.1, 0.2, 1.0], None) };

        unsafe {
            cmd.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: 1024.0,
                Height: 768.0,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }])
        };
        unsafe {
            cmd.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: 1024,
                bottom: 768,
            }])
        };

        unsafe { cmd.SetGraphicsRootSignature(&self.root_sig) };
        unsafe { cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST) };
        unsafe { cmd.IASetVertexBuffers(0, Some(&[self.vb_view])) };
        unsafe { cmd.DrawInstanced(3, 1, 0, 0) };

        unsafe {
            cmd.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(rt.clone())),
                        StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                        StateAfter: D3D12_RESOURCE_STATE_PRESENT,
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    }),
                },
            }])
        };

        unsafe { cmd.Close() }?;

        unsafe { self.queue.ExecuteCommandLists(&[Some(cmd.cast()?)]) };

        _ = unsafe { self.swapchain.Present(1, DXGI_PRESENT(0)) };

        unsafe { self.queue.Signal(&self.fence, self.fence_value) }?;

        if unsafe { self.fence.GetCompletedValue() } < self.fence_value {
            unsafe {
                self.fence
                    .SetEventOnCompletion(self.fence_value, self.fence_event)
            }?;
            unsafe { WaitForSingleObject(self.fence_event, INFINITE) };
        }
        self.fence_value += 1;

        Ok(())
    }
}
