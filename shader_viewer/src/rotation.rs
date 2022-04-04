
// Quaternions math (https://code.google.com/archive/p/kri/wikis/Quaternions.wiki)


use cgmath::{
	Vector3,
	Vector4,
	BaseNum
};


// fn quaternions_rotate<S: std::ops::Mul<Output = cgmath::Vector3<S>>>(q: Vector4<S>, v: Vector3<S>) -> Vector3<S> where S: BaseNum {
// 	let _q_xyz = Vector3::new(q.x, q.y, q.z);
// 	let _f = _q_xyz.cross(_q_xyz.cross(v) + v*q.w);
// 	Vector3::new(v.x+_f.x*new::<f32>(2.0), v.y+_f.y*new::<f32>(2.0), v.z+_f.z*new::<f32>(2.0))
// }

// pub fn rotate_vector<S: std::ops::Mul<Output = f32>>(v: Vector3<S>, axis: Vector3<S>, angle: S) -> Vector3<S> {
// 	let half_angle = (angle * new::<f32>(0.5)) * new<f32>(3.14159 / 180.0);
// 	let _a = axis * half_angle.sin();
// 	let qr = Vector4::new(_a.x, _a.y, _a.z, half_angle.cos());
// 	return quaternions_rotate(qr, v);
// }


fn quaternions_rotate(q: Vector4<f32>, v: Vector3<f32>) -> Vector3<f32>{
	let _q_xyz = Vector3::new(q.x, q.y, q.z);
	let _f = _q_xyz.cross(_q_xyz.cross(v) + v*q.w);
	Vector3::new(v.x+_f.x*2.0, v.y+_f.y*2.0, v.z+_f.z*2.0)
}

pub fn rotate_vector(v: Vector3<f32>, axis: Vector3<f32>, angle: f32) -> Vector3<f32> {
	let half_angle = (angle * 0.5) * 3.14159 / 180.0;
	let _a = axis * half_angle.sin();
	let qr = Vector4::new(_a.x, _a.y, _a.z, half_angle.cos());
	return quaternions_rotate(qr, v);
}

