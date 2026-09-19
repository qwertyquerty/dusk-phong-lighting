struct Uniforms {
    view_from_proj: mat4x4f,
    light_count: u32,
    enable_specular: u32,
    enable_rim: u32,
    specular_intensity: f32,
    rim_intensity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct GpuLight {
    position: vec3f,
    radius: f32,
    color: vec3f,
    _pad0: f32,
}

@group(0) @binding(0) var scene_depth: texture_2d<f32>;
@group(0) @binding(1) var scene_normal: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> lights: array<GpuLight>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var out: VertexOutput;
    let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
    out.position = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

fn view_position(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, depth, 1.0);
    let position = uniforms.view_from_proj * ndc;
    return position.xyz / position.w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let size = vec2<i32>(textureDimensions(scene_depth));
    let texel = clamp(vec2<i32>(in.uv * vec2f(size)), vec2<i32>(0i), size - 1i);
    let depth = textureLoad(scene_depth, texel, 0i).r;
    if depth <= 0.0 {
        return vec4f(0.0);
    }

    let normal_sample = textureLoad(scene_normal, texel, 0i);
    if normal_sample.a <= 0.0 {
        return vec4f(0.0);
    }
    let normal = normalize(normal_sample.rgb * 2.0 - 1.0);

    let view_pos = view_position(in.uv, depth);
    let view_dir = normalize(-view_pos);

    var added = vec3f(0.0);
    if uniforms.enable_specular != 0u {
        for (var i = 0u; i < uniforms.light_count; i++) {
            let light = lights[i];
            let to_light = light.position - view_pos;
            let dist = length(to_light);
            if dist < 0.0001 {
                continue;
            }
            var atten = 1.0;
            if light.radius > 0.0 {
                let falloff = saturate(1.0 - (dist * dist) / (light.radius * light.radius));
                atten = falloff * falloff;
                if atten <= 0.0 {
                    continue;
                }
            }
            let light_dir = to_light / dist;
            let n_dot_l = dot(normal, light_dir);
            if n_dot_l <= 0.0 {
                continue;
            }
            let h = normalize(light_dir + view_dir);
            let spec = pow(max(0.0, dot(normal, h)), 32.0);
            added += light.color * (spec * atten * uniforms.specular_intensity);
        }
    }
    if uniforms.enable_rim != 0u {
        let rim = pow(1.0 - max(0.0, dot(view_dir, normal)), 3.0) * uniforms.rim_intensity;
        added += vec3f(rim);
    }

    return vec4f(added, 0.0);
}
