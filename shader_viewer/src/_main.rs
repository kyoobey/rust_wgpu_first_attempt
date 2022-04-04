

use winit::{
	event::*,
	event_loop::{ControlFlow, EventLoop},
	window::{WindowBuilder, Window, Fullscreen}
};
use wgpu;
use wgpu::util::DeviceExt;
// use image;
use cgmath;
use std::time::{
	Instant,
	Duration
};


mod texture;
mod camera;
mod rotation;



// wgpu code

async fn run(event_loop: EventLoop<()>, window: Window) {

	let mut state = pollster::block_on(State::new(&window));
	let mut last_render_time = Instant::now();

	let fullscreen = Some(Fullscreen::Exclusive(
		event_loop
			.available_monitors().nth(0).expect("No available monitors")
			.video_modes().nth(0).expect("No available video modes")
	));

	event_loop.run(move |event, _, control_flow| match event {
		Event::DeviceEvent {
			event: DeviceEvent::MouseMotion{ delta, },
			..
		} => if state.mouse_pressed {
			state.camera_controller.process_mouse(delta.0, delta.1)
		},
		Event::WindowEvent {
			ref event,
			window_id,
		} if window_id == window.id() => {
			if !state.input(event) {
				match event {
					WindowEvent::CloseRequested
					| WindowEvent::KeyboardInput {
						input: KeyboardInput {
							state: ElementState::Pressed,
							virtual_keycode: Some(VirtualKeyCode::Escape),
							..
						},
						..
					} => *control_flow = ControlFlow::Exit,
					WindowEvent::KeyboardInput {
						input: KeyboardInput {
							state: ElementState::Pressed,
							virtual_keycode: Some(VirtualKeyCode::F),
							..
						},
						..
					} => {
						if window.fullscreen().is_some() {
							window.set_fullscreen(None);
						} else {
							window.set_fullscreen(
								fullscreen.clone()
							);
						}
					},
					WindowEvent::Resized(physical_size) => {
						state.resize(*physical_size);
					},
					WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
						state.resize(**new_inner_size);
					},
					_ => {}
				}
			}
		},
		Event::RedrawRequested(window_id) if window_id == window.id() => {
			let now = Instant::now();
			let dt = now - last_render_time;
			last_render_time = now;
			state.update(dt);
			match state.render() {
				Ok(_) => {},
				Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
				Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
				Err(e) => eprintln!("{:?}", e)
			}
		},
		Event::RedrawEventsCleared => {
			window.request_redraw();
		},
		_ => {}
	});

}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
	1.0, 0.0, 0.0, 0.0,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 0.5, 0.0,
	0.0, 0.0, 0.5, 1.0,
);


#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
	view_proj: [[f32; 4]; 4],
	eye: [f32; 3],
	_padding: f32
}

impl CameraUniform {
	fn new() -> Self {
		use cgmath::SquareMatrix;
		Self {
			view_proj: cgmath::Matrix4::identity().into(),
			eye: [0.0; 3],
			_padding: 0.0
		}
	}

	fn update(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
		// println!("{:?}", camera.position);
		self.eye = camera.position.into();
		// self.eye = camera.position.to_homogeneous().into();
		self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into();
		// self.view_proj = (camera.calc_matrix()).into();
	}
}


#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
	time: f32
}

impl Uniforms {
	fn new() -> Self {
		Self {
			time: 0.0
		}
	}

	fn update(&mut self, time: &Duration) {
		self.time = time.as_secs_f32();
	}
}



struct State {
	surface: wgpu::Surface,
	device: wgpu::Device,
	queue: wgpu::Queue,
	config: wgpu::SurfaceConfiguration,
	size: winit::dpi::PhysicalSize<u32>,
	clear_color: wgpu::Color,
	render_pipeline: wgpu::RenderPipeline,
	vertices: [Vertex; 4],
	vertex_buffer: wgpu::Buffer,
	num_vertices: u32,
	index_buffer: wgpu::Buffer,
	num_indices: u32,
	diffuse_bind_group: wgpu::BindGroup,
	diffuse_texture: texture::Texture,
	camera: camera::Camera,
	projection: camera::Projection,
	camera_controller: camera::CameraController,
	camera_uniform: CameraUniform,
	camera_buffer: wgpu::Buffer,
	camera_bind_group: wgpu::BindGroup,
	mouse_pressed: bool,
	time: Duration,
	uniforms: Uniforms,
	uniform_buffer: wgpu::Buffer,
	uniform_bind_group: wgpu::BindGroup
}

impl State {

	async fn new(window: &Window) -> Self {
		let size = window.inner_size();

		let instance = wgpu::Instance::new(wgpu::Backends::all());
		let surface = unsafe { instance.create_surface(window) };
		let adapter = instance.request_adapter(
			&wgpu::RequestAdapterOptions {
				// power_preference: wgpu::PowerPreference::default(),
				power_preference: wgpu::PowerPreference::HighPerformance,
				compatible_surface: Some(&surface),
				force_fallback_adapter: false
			}
		).await.unwrap();

		let (device, queue) = adapter.request_device(
			&wgpu::DeviceDescriptor {
				features: wgpu::Features::empty(),
				limits: wgpu::Limits::default(),
				label: None
			},
			None
		).await.unwrap();

		let config = wgpu::SurfaceConfiguration {
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			format: surface.get_preferred_format(&adapter).unwrap(),
			width: size.width,
			height: size.height,
			present_mode: wgpu::PresentMode::Fifo
		};
		surface.configure(&device, &config);

		let diffuse_bytes = include_bytes!("tree.png");
		// let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
		// let diffuse_rgba = diffuse_image.to_rgba8();

		// use image::GenericImageView;
		// let dimensions = diffuse_image.dimensions();

		// let texture_size = wgpu::Extent3d {
		// 	width: dimensions.0,
		// 	height: dimensions.1,
		// 	depth_or_array_layers: 1
		// };
		// let diffuse_texture = device.create_texture(
		// 	&wgpu::TextureDescriptor {
		// 		size: texture_size,
		// 		mip_level_count: 1,
		// 		sample_count: 1,
		// 		dimension: wgpu::TextureDimension::D2,
		// 		format: wgpu::TextureFormat::Rgba8UnormSrgb,
		// 		usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
		// 		label: Some("diffuse_texture")
		// 	}
		// );

		// queue.write_texture(
		// 	wgpu::ImageCopyTexture {
		// 		texture: &diffuse_texture,
		// 		mip_level: 0,
		// 		origin: wgpu::Origin3d::ZERO,
		// 		aspect: wgpu::TextureAspect::All,
		// 	},
		// 	&diffuse_rgba,
		// 	wgpu::ImageDataLayout {
		// 		offset: 0,
		// 		bytes_per_row: std::num::NonZeroU32::new(4 * dimensions.0),
		// 		rows_per_image: std::num::NonZeroU32::new(dimensions.1)
		// 	},
		// 	texture_size
		// );

		// let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
		// let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
		// 	address_mode_u: wgpu::AddressMode::Repeat,
		// 	address_mode_v: wgpu::AddressMode::Repeat,
		// 	address_mode_w: wgpu::AddressMode::Repeat,
		// 	mag_filter: wgpu::FilterMode::Linear,
		// 	min_filter: wgpu::FilterMode::Nearest,
		// 	mipmap_filter: wgpu::FilterMode::Nearest,
		// 	..Default::default()
		// });

		let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "tree.png").unwrap();

		let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Texture {
						multisampled: false,
						view_dimension: wgpu::TextureViewDimension::D2,
						sample_type: wgpu::TextureSampleType::Float { filterable: true }
					},
					count: None
				},
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
					count: None
				}
			],
			label: Some("texture_bind_group_layout")
		});

		let diffuse_bind_group = device.create_bind_group(
			&wgpu::BindGroupDescriptor {
				layout: &texture_bind_group_layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						// resource: wgpu::BindingResource::TextureView(&diffuse_texture_view)
						resource: wgpu::BindingResource::TextureView(&diffuse_texture.view)
					},
					wgpu::BindGroupEntry {
						binding: 1,
						// resource: wgpu::BindingResource::Sampler(&diffuse_sampler)
						resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler)
					}
				],
				label: Some("diffuse_bind_group")
			}
		);

		let clear_color = wgpu::Color::BLACK;

		let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
			label: Some("shader"),
			source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into())
		});

		let vertices = VERTICES;

		let vertex_buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: Some("Vertex Buffer"),
				contents: bytemuck::cast_slice(&vertices),
				usage: wgpu::BufferUsages::VERTEX
			}
		);
		let num_vertices = vertices.len() as u32;

		let index_buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: Some("Index Buffer"),
				contents: bytemuck::cast_slice(INDICES),
				usage: wgpu::BufferUsages::INDEX
			}
		);
		let num_indices = INDICES.len() as u32;

		// let camera = Camera {
		// 	eye: (0.0, 0.0, 3.0).into(),
		// 	target: (0.0, 0.0, 2.0).into(),
		// 	up: cgmath::Vector3::unit_y(),
		// 	aspect: config.width as f32 / config.height as f32,
		// 	fovy: 45.0,
		// 	znear: 0.1,
		// 	zfar: 100.0
		// };

		let camera = camera::Camera::new((3.0, 3.0, 3.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
		let projection = camera::Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);
		let camera_controller = camera::CameraController::new(20.0, 1.0);

		let mut camera_uniform = CameraUniform::new();
		camera_uniform.update(&camera, &projection);

		let camera_buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor{
				label: Some("Camera Buffer"),
				contents: bytemuck::cast_slice(&[camera_uniform]),
				usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
			}
		);

		let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: None
					},
					count: None
				}
			],
			label: Some("camera_bind_group_layout")
		});

		let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			layout: &camera_bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: camera_buffer.as_entire_binding()
				}
			],
			label: Some("camera_bind_group")
		});

		let camera_controller = camera::CameraController::new(4.0, 0.4);

		let time = Duration::from_secs(0);

		let mut uniforms = Uniforms::new();
		uniforms.update(&time);

		let uniform_buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor{
				label: Some("Camera Buffer"),
				contents: bytemuck::cast_slice(&[uniforms]),
				usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
			}	
		);

		let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: None
					},
					count: None
				}
			],
			label: Some("uniform_bind_group_layout")
		});

		let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			layout: &uniform_bind_group_layout,
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: uniform_buffer.as_entire_binding()
				}
			],
			label: Some("uniform_bind_group")
		});

		let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("Render Pipeline Layout"),
			bind_group_layouts: &[
				&texture_bind_group_layout,
				&camera_bind_group_layout,
				&uniform_bind_group_layout
			],
			push_constant_ranges: &[]
		});

		let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("Render Pipeline"),
			layout: Some(&render_pipeline_layout),
			vertex: wgpu::VertexState {
				module: &shader,
				entry_point: "vs_main",
				buffers: &[
					Vertex::desc()
				]
			},
			fragment: Some(wgpu::FragmentState {
				module: &shader,
				entry_point: "fs_main",
				targets: &[wgpu::ColorTargetState {
					format: config.format,
					blend: Some(wgpu::BlendState::REPLACE),
					write_mask: wgpu::ColorWrites::ALL
				}]
			}),
			primitive: wgpu::PrimitiveState {
				topology: wgpu::PrimitiveTopology::TriangleList,
				strip_index_format: None,
				front_face: wgpu::FrontFace::Ccw,
				cull_mode: Some(wgpu::Face::Back),
				polygon_mode: wgpu::PolygonMode::Fill,
				unclipped_depth: false,
				conservative: false
			},
			depth_stencil: None,
			multisample: wgpu::MultisampleState {
				count: 1,
				mask: 10,
				alpha_to_coverage_enabled: false
			},
			multiview: None
		});


		Self {
			surface, device, queue, config, size, clear_color, render_pipeline,
			vertices, vertex_buffer, num_vertices, index_buffer, num_indices,
			diffuse_bind_group, diffuse_texture,
			camera, projection, camera_uniform, camera_buffer, camera_bind_group, camera_controller,
			mouse_pressed: false,
			uniforms, uniform_bind_group, uniform_buffer, time
		}
	}

	fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
		if new_size.width > 0 && new_size.height > 0 {
			self.size = new_size;
			self.config.width = new_size.width;
			self.config.height = new_size.height;
			self.surface.configure(&self.device, &self.config);
			self.projection.resize(new_size.width, new_size.height);
		}
	}

	fn input(&mut self, event: &WindowEvent) -> bool {
		match event {
			WindowEvent::CursorMoved {
				position: winit::dpi::PhysicalPosition { x, y },
				..
			} => {
				// println!("{}, {}", x, y);
				self.clear_color = wgpu::Color {
					r: *x / self.size.width as f64,
					g: *y / self.size.height as f64,
					b: 0.1,
					a: 1.0
				};
				true
			},
			WindowEvent::KeyboardInput {
				input:
				KeyboardInput {
					virtual_keycode: Some(key),
					state,
					..
					},
				..
			} => self.camera_controller.process_keyboard(*key, *state),
			WindowEvent::MouseWheel { delta, .. } => {
				self.camera_controller.process_scroll(delta);
				true
			}
			WindowEvent::MouseInput {
				button: MouseButton::Left,
				state,
				..
			} => {
				self.mouse_pressed = *state == ElementState::Pressed;
				true
			},
			_ => false
		}
	}

	fn update(&mut self, dt: std::time::Duration) {
		self.camera_controller.update_camera(&mut self.camera, dt);
		self.camera_uniform.update(&self.camera, &self.projection);
		self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
		self.time += dt;
		self.uniforms.update(&self.time);
		self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));



		use cgmath::Transform;
		use cgmath::InnerSpace;
		use cgmath::EuclideanSpace;

		let inv = self.projection.calc_matrix().inverse_transform().unwrap();
		let _vertex_0 = inv * cgmath::Vector4::from([-1.0, -1.0, 0.0, 1.0]);
		let _vertex_1 = inv * cgmath::Vector4::from([-1.0,  1.0, 0.0, 1.0]);
		let _vertex_2 = inv * cgmath::Vector4::from([ 1.0,  1.0, 0.0, 1.0]);
		let _vertex_3 = inv * cgmath::Vector4::from([ 1.0, -1.0, 0.0, 1.0]);

		let _vertex_0 = cgmath::Vector3::new(_vertex_0.x, _vertex_0.y, _vertex_0.z);
		let _vertex_1 = cgmath::Vector3::new(_vertex_1.x, _vertex_1.y, _vertex_1.z);
		let _vertex_2 = cgmath::Vector3::new(_vertex_2.x, _vertex_2.y, _vertex_2.z);
		let _vertex_3 = cgmath::Vector3::new(_vertex_3.x, _vertex_3.y, _vertex_3.z);

		use std::f32::consts::PI;

		let yaw_deg = self.camera.yaw.0 * (180.0/PI);
		let pitch_deg = self.camera.pitch.0 * (180.0/PI);

		let unit_y = cgmath::Vector3::new(0.,1.,0.);
		let cam = self.camera.position;
		let cam = cgmath::Vector3::new(cam.x, cam.y, cam.z);

		let _vertex_0 = cam + rotation::rotate_vector(_vertex_0 - cam, unit_y, -yaw_deg-90.);
		let _vertex_1 = cam + rotation::rotate_vector(_vertex_1 - cam, unit_y, -yaw_deg-90.);
		let _vertex_2 = cam + rotation::rotate_vector(_vertex_2 - cam, unit_y, -yaw_deg-90.);
		let _vertex_3 = cam + rotation::rotate_vector(_vertex_3 - cam, unit_y, -yaw_deg-90.);

		let pitch_rot_dir = unit_y.cross((_vertex_1 - _vertex_0).normalize().cross((_vertex_3 - _vertex_0).normalize()));
		let _vertex_0 = cam + rotation::rotate_vector(_vertex_0 - cam, pitch_rot_dir, -pitch_deg);
		let _vertex_1 = cam + rotation::rotate_vector(_vertex_1 - cam, pitch_rot_dir, -pitch_deg);
		let _vertex_2 = cam + rotation::rotate_vector(_vertex_2 - cam, pitch_rot_dir, -pitch_deg);
		let _vertex_3 = cam + rotation::rotate_vector(_vertex_3 - cam, pitch_rot_dir, -pitch_deg);

		// println!("before {:?}", _vertex_0);
		let cam = cgmath::Vector3::new(cam.x, cam.y, cam.z);
		let centroid = cgmath::Vector3::new((_vertex_0.x + _vertex_2.x) / 2., (_vertex_0.y + _vertex_2.y) / 2., (_vertex_0.z + _vertex_2.z) / 2.);
		let _vertex_0 = _vertex_0 + cam - centroid;
		let _vertex_1 = _vertex_1 + cam - centroid;
		let _vertex_2 = _vertex_2 + cam - centroid;
		let _vertex_3 = _vertex_3 + cam - centroid;
		// println!("after {:?}", _vertex_0);

		// println!("before {:?}", _vertex_0);

		let _vertex_0 = _vertex_0 + self.camera.look_dir();
		let _vertex_1 = _vertex_1 + self.camera.look_dir();
		let _vertex_2 = _vertex_2 + self.camera.look_dir();
		let _vertex_3 = _vertex_3 + self.camera.look_dir();

		// println!("after {:?}", _vertex_0);

		self.vertices[0].position = [_vertex_0.x, _vertex_0.y, _vertex_0.z];
		self.vertices[1].position = [_vertex_1.x, _vertex_1.y, _vertex_1.z];
		self.vertices[2].position = [_vertex_2.x, _vertex_2.y, _vertex_2.z];
		self.vertices[3].position = [_vertex_3.x, _vertex_3.y, _vertex_3.z];

		self.vertex_buffer = self.device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: Some("Vertex Buffer"),
				contents: bytemuck::cast_slice(&self.vertices),
				usage: wgpu::BufferUsages::VERTEX
			}
		);


		// println!("{:?}", self.uniforms.time);
	}

	fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
		let output = self.surface.get_current_texture()?;
		let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("Render Encoder")
		});

		{
			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("Render Pass"),
				color_attachments: &[wgpu::RenderPassColorAttachment {
					view: &view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(
							// wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 }
							self.clear_color
						),
						store: true
					}
				}],
				depth_stencil_attachment: None
			});

			render_pass.set_pipeline(&self.render_pipeline);
			render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
			render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
			render_pass.set_bind_group(2, &self.uniform_bind_group, &[]);
			render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
			render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
			// render_pass.draw(0..self.num_vertices, 0..1);
			render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
		}

		self.queue.submit(std::iter::once(encoder.finish()));
		output.present();

		Ok(())
	}


}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
	position: [f32; 3],
	// color: [f32; 3]
	// tex_coords: [f32; 2]
}

impl Vertex {

	// const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

	fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
		wgpu::VertexBufferLayout {
			array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
			step_mode: wgpu::VertexStepMode::Vertex,
			attributes: &[
				wgpu::VertexAttribute {
					offset: 0,
					shader_location: 0,
					format: wgpu::VertexFormat::Float32x3
				},
				// wgpu::VertexAttribute {
				// 	offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
				// 	shader_location: 1,
				// 	// format: wgpu::VertexFormat::Float32x3
				// 	format: wgpu::VertexFormat::Float32x2
				// }
			]
			// attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3]
		}
	}
}


// const VERTICES: &[Vertex] = &[
// 	Vertex { position: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0] },
// 	Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },
// 	Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] }
// ];

// const VERTICES: &[Vertex] = &[
// 	Vertex { position: [-0.0868241, 0.49240386, 0.0], color: [0.5, 0.0, 0.5] },
// 	Vertex { position: [-0.49513406, 0.06958647, 0.0], color: [1.0, 0.0, 0.0] },
// 	Vertex { position: [-0.21918549, -0.44939706, 0.0], color: [0.5, 0.0, 0.5] },
// 	Vertex { position: [0.35966998, -0.3473291, 0.0], color: [0.5, 0.0, 0.5] },
// 	Vertex { position: [0.44147372, 0.2347359, 0.0], color: [0.5, 0.0, 0.5] }
// ];

// const VERTICES: &[Vertex] = &[
// 	Vertex { position: [-0.0868241, 0.49240386, 0.0], tex_coords: [0.4131759, 0.00759614], },
// 	Vertex { position: [-0.49513406, 0.06958647, 0.0], tex_coords: [0.0048659444, 0.43041354], },
// 	Vertex { position: [-0.21918549, -0.44939706, 0.0], tex_coords: [0.28081453, 0.949397], },
// 	Vertex { position: [0.35966998, -0.3473291, 0.0], tex_coords: [0.85967, 0.84732914], },
// 	Vertex { position: [0.44147372, 0.2347359, 0.0], tex_coords: [0.9414737, 0.2652641], }
// ];
// const VERTICES: &[Vertex] = &[
// 	Vertex { position: [-0.0868241, 0.49240386, 0.0], tex_coords: [0.4131759, 0.00759614], },
// 	Vertex { position: [-0.49513406, 0.06958647, 0.0], tex_coords: [0.0048659444, 0.43041354], },
// 	Vertex { position: [-0.21918549, -0.44939706, 0.0], tex_coords: [0.28081453, 0.949397], },
// 	Vertex { position: [0.35966998, -0.3473291, 0.0], tex_coords: [0.85967, 0.84732914], },
// 	Vertex { position: [0.44147372, 0.2347359, 0.0], tex_coords: [0.9414737, 0.2652641], }
// ];

const VERTICES: [Vertex; 4] = [
	Vertex { position: [-1.0, -1.0, 0.0] },
	Vertex { position: [-1.0,  1.0, 0.0] },
	Vertex { position: [ 1.0,  1.0, 0.0] },
	Vertex { position: [ 1.0, -1.0, 0.0] },
];

const INDICES: &[u16] = &[
	2, 1, 0,
	3, 2, 0
];

// const INDICES: &[u16] = &[
// 	0, 1, 4,
// 	1, 2, 4,
// 	2, 3, 4
// ];








fn main() {
// 	println!("\n\n=======================================================
// shader_viewer: A Program to view your shaders
// =======================================================\n\n");



	// wgpu code starts here

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new().build(&event_loop).unwrap();


	// enable logging and run shader

	#[cfg(not(target_arch = "wasm32"))]
	{
		env_logger::init();
		pollster::block_on(run(event_loop, window));
	}
	#[cfg(target_arch = "wasm32")]
	{
		use console_error_panic_hook;
		use console_log;
		use web_sys;
		use winit::platform::web::WindowExtWebSys;

		std::panic::set_hook(Box::new(console_error_panic_hook::hook));
		console_log::init().expect("could not initialize logger");

		// on wasm, append the canvas to the document body
		web_sys::window()
			.and_then(|win| win.document())
			.and_then(|doc| doc.body())
			.and_then(|body| {
				body.append_child(&web_sys::Element::from(window.canvas()))
					.ok()
			})
			.expect("couldn't append canvas to document body");
		wasm_bindgen_futures::spawn_local(run(event_loop, window));
	}


	// env_logger::init();
	// pollster::block_on(run(event_loop, window));

}

