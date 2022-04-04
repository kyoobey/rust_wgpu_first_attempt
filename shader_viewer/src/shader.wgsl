//******************************************************
// wasm

struct CameraUniform {
	view_proj: mat4x4<f32>,
	eye: vec3<f32>
};

@group(1)
@binding(0)
var<uniform> camera: CameraUniform;

struct Uniforms {
	time: f32
};

@group(2)
@binding(0)
var<uniform> uniforms: Uniforms;


// vertex shader


struct VertexInput {
	@location(0) position: vec3<f32>
};

struct VertexOutput {
	@builtin(position) clip_position: vec4<f32>,
	@location(0) tex_coords: vec2<f32>,
	@location(1) position: vec3<f32>,
	@location(2) camera_direction: vec3<f32>
};

@stage(vertex)
fn vs_main(
	model: VertexInput
) -> VertexOutput {
	var out: VertexOutput;

	// let x = f32(1 - i32(in_vertex_index)) * 0.5;
	// let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
	// out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
	// return out;

	// out.color = model.color;
	// out.tex_coords = model.tex_coords;

	out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);

	out.position = model.position;
	out.camera_direction = normalize(model.position - camera.eye);

	return out;
}









// ray marching code

fn quaternions_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32>{
	return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}

fn rotate_vector(v: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
	var half_angle: f32 = (angle * 0.5) * 3.14159 / 180.0;
	var qr: vec4<f32> = vec4<f32>(axis * sin(half_angle), cos(half_angle));
	return quaternions_rotate(qr, v);
}


fn sdf_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
	var q = abs(p) - b;
	return length(max(q,vec3<f32>(0.0))) + min(max(q.x,max(q.y,q.z)),0.0);
}

fn modifier_finite_repeat(p: vec3<f32>, k: vec3<f32>, lima: vec3<f32>, limb: vec3<f32>) -> vec3<f32> {
	// lima: limit start vector ( vec3(-1.) )
	// limb: limit end vector ( vec3(1.) )
	return p - k*clamp(round(p/k),lima,limb);
}

let PI: f32 = 3.141592653589793238;

fn map (value: f32, min1: f32, max1: f32, min2: f32, max2: f32) -> f32 {
	return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

fn sdf_mandelbulb(p: vec3<f32>) -> f32 {
	var z = p;
	var dr: f32 = 1.0;
	var r: f32 = 0.0;
	var iterations: i32 = 0;
	var power: f32 = 8.0 + (5.0 * map (sin (uniforms.time * PI / 10.0 + PI), -1.0, 1.0, 0.0, 1.0));

	for (var i: i32 = 0; i < 2; i=i+1) {
		var iterations = i;
		var r = length (z);

		if (r > 2.0) {
			break;
		}

		// convert to polar coordinates
		var theta = acos (z.z / r);
		var phi = atan2 (z.y , z.x);
		var dr = pow (r, power - 1.0) * power * dr + 1.0;

		// scale and rotate the point
		var zr = pow (r, power);
		theta = theta * power;
		phi = phi * power;

		// convert back to cartesian coordinates
		z = zr * vec3<f32> (sin (theta) * cos (phi), sin (phi) * sin (theta), cos (theta));
		z = z + p;
	}
	var dst: f32 = 0.5 * log (r) * r / dr;
	return dst;
}

// http://iquilezles.org/www/articles/mandelbulb/mandelbulb.htm
fn sdf_mandelbulb_2(p: vec3<f32>) -> f32 {
	var w: vec3<f32> = p;
	var m: f32 = dot(w,w);

	var trap: vec4<f32> = vec4<f32>(abs(w),m);
	var dz: f32 = 1.0;
	
	for( var i: i32 = 0; i < 2; i=i+1 ) {
		dz = 8.0*pow(m,3.5)*dz + 1.0;

		// var m2: f32 = m*m;
		// var m4: f32 = m2*m2;
		// dz = 8.0*sqrt(m4*m2*m)*dz + 1.0;

		// var x: f32 = w.x; var x2: f32 = x*x; var x4: f32 = x2*x2;
		// var y: f32 = w.y; var y2: f32 = y*y; var y4: f32 = y2*y2;
		// var z: f32 = w.z; var z2: f32 = z*z; var z4: f32 = z2*z2;

		// var k3: f32 = x2 + z2;
		// var k2: f32 = 1. / sqrt( k3*k3*k3*k3*k3*k3*k3 );
		// var k1: f32 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;
		// var k4: f32 = x2 - y2 + z2;

		// w.x = p.x +  64.0*x*y*z*(x2-z2)*k4*(x4-6.0*x2*z2+z4)*k1*k2;
		// w.y = p.y + -16.0*y2*k3*k4*k4 + k1*k1;
		// w.z = p.z +  -8.0*y*k4*(x4*x4 - 28.0*x4*x2*z2 + 70.0*x4*z4 - 28.0*x2*z2*z4 + z4*z4)*k1*k2;

		// z = z^8+z
		var r: f32 = length(w);
		var b: f32 = 8.0*acos( w.y/r);
		var a: f32 = 8.0*atan2( w.x, w.z );
		w = p + pow(r,8.0) * vec3<f32>( sin(b)*sin(a), cos(b), sin(b)*cos(a) );

		trap = min( trap, vec4<f32>(abs(w),m) );

		m = dot(w,w);
		if( m > 256.0 ) {
			break;
		}
	}

	// resColor = vec4(m,trap.yzw);

	// distance estimation (through the Hubbard-Douady potential)
	return 0.25*log(m)*sqrt(m)/dz;
}

fn SDF(_p: vec3<f32>) -> f32 {
	// return sdf_mandelbulb_2(p - vec3<f32>(0., 0., -2.));
	// return sdf_box(p - vec3<f32>(0., 0., -10.), vec3<f32>(0.1));
	// return sdf_box(p, vec3<f32>((sin(uniforms.time)*0.5+0.5)+1.));
	var p: vec3<f32> = modifier_finite_repeat(_p, vec3<f32>(sin(uniforms.time)*3.+3.), vec3<f32>(-2.), vec3<f32>(2.));
	return sdf_box(
		rotate_vector(p, vec3<f32>(0.1), 90.*sin(uniforms.time)),
		vec3<f32>((sin(uniforms.time)*0.5+0.5)+1.)
	);
	// return length(p) - 0.1;
	// return 0.0;
}


struct MarchOutput {
	steps: i32,
	depth: f32,
	minimum_distance: f32,
	hit: bool
};

fn march(
	point: vec3<f32>, direction: vec3<f32>,
	max_steps: i32, max_shading_distance: f32, min_hit_distance: f32
) -> MarchOutput {
	var out = MarchOutput ( 0, 0.0, max_shading_distance, false );

	for (out.steps=0; out.depth < max_shading_distance && out.steps < max_steps; out.steps=out.steps+1) {
		var current_position: vec3<f32> = point + direction * out.depth;
		var current_distance: f32 = SDF(current_position);

		if (abs(current_distance) < min_hit_distance) {
			out.hit = true;
			break;
		}

		out.minimum_distance = min(out.minimum_distance, current_distance);
		out.depth = out.depth + current_distance;
	}

	return out;
}






// fragment shader


@group(0)
@binding(0)
var t_diffuse: texture_2d<f32>;

@group(0)
@binding(1)
var s_diffuse: sampler;


@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	// return vec4<f32>(0.3, 0.2, 0.1, 1.0);

	// return vec4<f32>(in.color, 1.0);

	// return textureSample(t_diffuse, s_diffuse, in.tex_coords);

	// return vec4<f32>(camera.eye, 1.0);
	// return vec4<f32>(vec3<f32>(sin(uniforms.time)), 1.0);



	var scene_data: MarchOutput = march(
		camera.eye, in.camera_direction,
		1023, 1000., 0.001
	);



	return vec4<f32>(vec3<f32>(f32(scene_data.steps)/1023.), 1.0);
	// if (scene_data.hit) {
	// 	return vec4<f32>(1.0, 0.0, 0.0, 1.0);
	// } else {
	// 	return vec4<f32>(0.0, 0.0, 0.0, 1.0);
	// }
}

//******************************************************

// native


// struct CameraUniform {
// 	view_proj: mat4x4<f32>;
// 	eye: vec3<f32>;
// };

// [[group(1), binding(0)]]
// var<uniform> camera: CameraUniform;

// struct Uniforms {
// 	time: f32;
// };

// [[group(2), binding(0)]]
// var<uniform> uniforms: Uniforms;


// // vertex shader


// struct VertexInput {
// 	[[location(0)]] position: vec3<f32>;
// };

// struct VertexOutput {
// 	[[builtin(position)]] clip_position: vec4<f32>;
// 	[[location(0)]] tex_coords: vec2<f32>;
// 	[[location(1)]] position: vec3<f32>;
// 	[[location(2)]] camera_direction: vec3<f32>;
// };

// [[stage(vertex)]]
// fn vs_main(
// 	model: VertexInput
// ) -> VertexOutput {
// 	var out: VertexOutput;

// 	// let x = f32(1 - i32(in_vertex_index)) * 0.5;
// 	// let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
// 	// out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
// 	// return out;

// 	// out.color = model.color;
// 	// out.tex_coords = model.tex_coords;

// 	out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);

// 	out.position = model.position;
// 	out.camera_direction = normalize(model.position - camera.eye);

// 	return out;
// }









// // ray marching code

// fn quaternions_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32>{
// 	return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
// }

// fn rotate_vector(v: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
// 	var half_angle: f32 = (angle * 0.5) * 3.14159 / 180.0;
// 	var qr: vec4<f32> = vec4<f32>(axis * sin(half_angle), cos(half_angle));
// 	return quaternions_rotate(qr, v);
// }


// fn sdf_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
// 	var q = abs(p) - b;
// 	return length(max(q,vec3<f32>(0.0))) + min(max(q.x,max(q.y,q.z)),0.0);
// }

// fn modifier_finite_repeat(p: vec3<f32>, k: vec3<f32>, lima: vec3<f32>, limb: vec3<f32>) -> vec3<f32> {
// 	// lima: limit start vector ( vec3(-1.) )
// 	// limb: limit end vector ( vec3(1.) )
// 	return p - k*clamp(round(p/k),lima,limb);
// }

// let PI: f32 = 3.141592653589793238;

// fn map (value: f32, min1: f32, max1: f32, min2: f32, max2: f32) -> f32 {
// 	return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
// }

// fn sdf_mandelbulb(p: vec3<f32>) -> f32 {
// 	var z = p;
// 	var dr: f32 = 1.0;
// 	var r: f32 = 0.0;
// 	var iterations: i32 = 0;
// 	var power: f32 = 8.0 + (5.0 * map (sin (uniforms.time * PI / 10.0 + PI), -1.0, 1.0, 0.0, 1.0));

// 	for (var i: i32 = 0; i < 2; i=i+1) {
// 		var iterations = i;
// 		var r = length (z);

// 		if (r > 2.0) {
// 			break;
// 		}

// 		// convert to polar coordinates
// 		var theta = acos (z.z / r);
// 		var phi = atan2 (z.y , z.x);
// 		var dr = pow (r, power - 1.0) * power * dr + 1.0;

// 		// scale and rotate the point
// 		var zr = pow (r, power);
// 		theta = theta * power;
// 		phi = phi * power;

// 		// convert back to cartesian coordinates
// 		z = zr * vec3<f32> (sin (theta) * cos (phi), sin (phi) * sin (theta), cos (theta));
// 		z = z + p;
// 	}
// 	var dst: f32 = 0.5 * log (r) * r / dr;
// 	return dst;
// }

// // http://iquilezles.org/www/articles/mandelbulb/mandelbulb.htm
// fn sdf_mandelbulb_2(p: vec3<f32>) -> f32 {
// 	var w: vec3<f32> = p;
// 	var m: f32 = dot(w,w);

// 	var trap: vec4<f32> = vec4<f32>(abs(w),m);
// 	var dz: f32 = 1.0;
	
// 	for( var i: i32 = 0; i < 2; i=i+1 ) {
// 		dz = 8.0*pow(m,3.5)*dz + 1.0;

// 		// var m2: f32 = m*m;
// 		// var m4: f32 = m2*m2;
// 		// dz = 8.0*sqrt(m4*m2*m)*dz + 1.0;

// 		// var x: f32 = w.x; var x2: f32 = x*x; var x4: f32 = x2*x2;
// 		// var y: f32 = w.y; var y2: f32 = y*y; var y4: f32 = y2*y2;
// 		// var z: f32 = w.z; var z2: f32 = z*z; var z4: f32 = z2*z2;

// 		// var k3: f32 = x2 + z2;
// 		// var k2: f32 = 1. / sqrt( k3*k3*k3*k3*k3*k3*k3 );
// 		// var k1: f32 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;
// 		// var k4: f32 = x2 - y2 + z2;

// 		// w.x = p.x +  64.0*x*y*z*(x2-z2)*k4*(x4-6.0*x2*z2+z4)*k1*k2;
// 		// w.y = p.y + -16.0*y2*k3*k4*k4 + k1*k1;
// 		// w.z = p.z +  -8.0*y*k4*(x4*x4 - 28.0*x4*x2*z2 + 70.0*x4*z4 - 28.0*x2*z2*z4 + z4*z4)*k1*k2;

// 		// z = z^8+z
// 		var r: f32 = length(w);
// 		var b: f32 = 8.0*acos( w.y/r);
// 		var a: f32 = 8.0*atan2( w.x, w.z );
// 		w = p + pow(r,8.0) * vec3<f32>( sin(b)*sin(a), cos(b), sin(b)*cos(a) );

// 		trap = min( trap, vec4<f32>(abs(w),m) );

// 		m = dot(w,w);
// 		if( m > 256.0 ) {
// 			break;
// 		}
// 	}

// 	// resColor = vec4(m,trap.yzw);

// 	// distance estimation (through the Hubbard-Douady potential)
// 	return 0.25*log(m)*sqrt(m)/dz;
// }

// fn SDF(_p: vec3<f32>) -> f32 {
// 	// return sdf_mandelbulb_2(p - vec3<f32>(0., 0., -2.));
// 	// return sdf_box(p - vec3<f32>(0., 0., -10.), vec3<f32>(0.1));
// 	// return sdf_box(p, vec3<f32>((sin(uniforms.time)*0.5+0.5)+1.));
// 	var p: vec3<f32> = modifier_finite_repeat(_p, vec3<f32>(sin(uniforms.time)*3.+3.), vec3<f32>(-2.), vec3<f32>(2.));
// 	return sdf_box(
// 		rotate_vector(p, vec3<f32>(0.1), 90.*sin(uniforms.time)),
// 		vec3<f32>((sin(uniforms.time)*0.5+0.5)+1.)
// 	);
// 	// return length(p) - 0.1;
// 	// return 0.0;
// }


// struct MarchOutput {
// 	steps: i32;
// 	depth: f32;
// 	minimum_distance: f32;
// 	hit: bool;
// };

// fn march(
// 	point: vec3<f32>, direction: vec3<f32>,
// 	max_steps: i32, max_shading_distance: f32, min_hit_distance: f32
// ) -> MarchOutput {
// 	var out = MarchOutput ( 0, 0.0, max_shading_distance, false );

// 	for (out.steps=0; out.depth < max_shading_distance && out.steps < max_steps; out.steps=out.steps+1) {
// 		var current_position: vec3<f32> = point + direction * out.depth;
// 		var current_distance: f32 = SDF(current_position);

// 		if (abs(current_distance) < min_hit_distance) {
// 			out.hit = true;
// 			break;
// 		}

// 		out.minimum_distance = min(out.minimum_distance, current_distance);
// 		out.depth = out.depth + current_distance;
// 	}

// 	return out;
// }






// // fragment shader


// [[group(0), binding(0)]]
// var t_diffuse: texture_2d<f32>;

// [[group(0), binding(1)]]
// var s_diffuse: sampler;


// [[stage(fragment)]]
// fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
// 	// return vec4<f32>(0.3, 0.2, 0.1, 1.0);

// 	// return vec4<f32>(in.color, 1.0);

// 	// return textureSample(t_diffuse, s_diffuse, in.tex_coords);

// 	// return vec4<f32>(camera.eye, 1.0);
// 	// return vec4<f32>(vec3<f32>(sin(uniforms.time)), 1.0);



// 	var scene_data: MarchOutput = march(
// 		camera.eye, in.camera_direction,
// 		1023, 1000., 0.001
// 	);



// 	return vec4<f32>(vec3<f32>(f32(scene_data.steps)/1023.), 1.0);
// 	// if (scene_data.hit) {
// 	// 	return vec4<f32>(1.0, 0.0, 0.0, 1.0);
// 	// } else {
// 	// 	return vec4<f32>(0.0, 0.0, 0.0, 1.0);
// 	// }
// }