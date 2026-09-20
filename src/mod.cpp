#include "JSystem/JMath/JMath.h"
#include "SSystem/SComponent/c_xyz.h"
#include "d/d_com_inf_game.h"
#include "d/d_kankyo.h"
#include "mods/service.hpp"
#include "mods/svc/camera.h"
#include "mods/svc/config.h"
#include "mods/svc/gfx.h"
#include "mods/svc/hook.h"
#include "mods/svc/hook.hpp"
#include "mods/svc/log.h"
#include "mods/svc/resource.h"
#include "mods/svc/ui.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <webgpu/webgpu.h>

DEFINE_MOD();
IMPORT_SERVICE(LogService, svc_log);
IMPORT_SERVICE(ConfigService, svc_config);
IMPORT_SERVICE(ResourceService, svc_resource);
IMPORT_SERVICE(UiService, svc_ui);
IMPORT_SERVICE(GfxService, svc_gfx);
IMPORT_SERVICE(CameraService, svc_camera);
IMPORT_SERVICE(HookService, svc_hook);

namespace {

DEFINE_HOOK(dKy_setLight_nowroom_actor, SetLightNowroomActor);

ConfigVarHandle g_cvarEnabled = 0;
ConfigVarHandle g_cvarEnableSpecular = 0;
ConfigVarHandle g_cvarEnableRim = 0;
ConfigVarHandle g_cvarSpecularPct = 0;
ConfigVarHandle g_cvarRimPct = 0;
ConfigVarHandle g_cvarAmbientPct = 0;
ConfigVarHandle g_cvarDiffusePct = 0;

GfxDrawTypeHandle g_drawType = 0;
GfxStageHookHandle g_afterOpaqueHook = 0;

ResourceBuffer g_shaderSource = RESOURCE_BUFFER_INIT;
GfxDeviceInfo g_deviceInfo = GFX_DEVICE_INFO_INIT;
GfxRenderTargetLayout g_sceneTargetLayout = GFX_RENDER_TARGET_LAYOUT_INIT;
WGPURenderPipeline g_compositePipeline = nullptr;
WGPUBindGroupLayout g_compositeLayout = nullptr;

constexpr uint32_t kMaxLights = 128;

struct PhongUniforms {
    float view_from_proj[16];
    uint32_t light_count;
    uint32_t enable_specular;
    uint32_t enable_rim;
    float specular_intensity;
    float rim_intensity;
    float _pad0;
    float _pad1;
    float _pad2;
};
static_assert(sizeof(PhongUniforms) % 16 == 0);

struct GpuLight {
    float position[3];
    float radius;
    float color[3];
    float _pad0;
};
static_assert(sizeof(GpuLight) == 32);

struct DrawPayload {
    WGPUTextureView sceneDepth;
    WGPUTextureView sceneNormal;
    uint32_t uniform_offset;
    uint32_t uniform_size;
    uint32_t storage_offset;
    uint32_t storage_size;
};
static_assert(sizeof(DrawPayload) <= GFX_INLINE_DRAW_PAYLOAD_SIZE);
static_assert(std::is_trivially_copyable_v<DrawPayload>);

int64_t get_int_option(ConfigVarHandle handle, int64_t fallback) {
    int64_t value = fallback;
    if (handle == 0 || svc_config->get_int(mod_ctx, handle, &value) != MOD_OK) {
        return fallback;
    }
    return value;
}

bool get_bool_option(ConfigVarHandle handle, bool fallback) {
    bool value = fallback;
    if (handle == 0 || svc_config->get_bool(mod_ctx, handle, &value) != MOD_OK) {
        return fallback;
    }
    return value;
}

float gx_color_channel(s16 value) {
    return std::clamp(static_cast<float>(value) / 255.0f, 0.0f, 4.0f);
}

void add_light(GpuLight* lights, uint32_t& count, const cXyz& position, const GXColorS10& color,
    float radius) {
    if (count >= kMaxLights) {
        return;
    }
    GpuLight& light = lights[count++];
    light.position[0] = position.x;
    light.position[1] = position.y;
    light.position[2] = position.z;
    light.radius = radius;
    light.color[0] = gx_color_channel(color.r);
    light.color[1] = gx_color_channel(color.g);
    light.color[2] = gx_color_channel(color.b);
    light._pad0 = 0.0f;
}

uint32_t gather_lights(GpuLight* lights) {
    dScnKy_env_light_c* env = dKy_getEnvlight();
    if (env == nullptr) {
        return 0;
    }

    uint32_t count = 0;
    add_light(lights, count, env->base_light.mPosition, env->base_light.mColor, 0.0f);

    for (const LIGHT_INFLUENCE* light : env->pointlight) {
        if (light != nullptr && light->mPow > 0.01f) {
            add_light(lights, count, light->mPosition, light->mColor, light->mPow);
        }
    }
    for (const LIGHT_INFLUENCE* light : env->efplight) {
        if (light != nullptr && light->mPow > 0.01f) {
            add_light(lights, count, light->mPosition, light->mColor, light->mPow);
        }
    }
    for (const LIGHT_INFLUENCE& light : env->bgparts_active_light) {
        if (light.mIndex != 0 && light.mPow != 0.0f) {
            add_light(lights, count, light.mPosition, light.mColor, light.mPow);
        }
    }
    return count;
}

template <class T>
T scale_color_channel(T value, float multiplier) {
    return static_cast<T>(std::clamp(static_cast<float>(value) * multiplier, 0.0f, 255.0f));
}

void on_set_light_nowroom_actor_post(ModContext*, void* args, void*, void*) {
    if (!get_bool_option(g_cvarEnabled, false)) {
        return;
    }
    dKy_tevstr_c* tevstr_p = mods::arg<dKy_tevstr_c*>(args, 0);
    if (tevstr_p == nullptr) {
        return;
    }

    const float ambientMultiplier =
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarAmbientPct, 100), 0, 300)) /
        100.0f;
    const float diffuseMultiplier =
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarDiffusePct, 100), 0, 300)) /
        100.0f;

    if (ambientMultiplier != 1.0f) {
        tevstr_p->AmbCol.r = scale_color_channel(tevstr_p->AmbCol.r, ambientMultiplier);
        tevstr_p->AmbCol.g = scale_color_channel(tevstr_p->AmbCol.g, ambientMultiplier);
        tevstr_p->AmbCol.b = scale_color_channel(tevstr_p->AmbCol.b, ambientMultiplier);
    }
    if (diffuseMultiplier != 1.0f) {
        for (J3DLightObj& lightObj : tevstr_p->mLights) {
            J3DLightInfo* light = lightObj.getLightInfo();
            light->mColor.r = scale_color_channel(light->mColor.r, diffuseMultiplier);
            light->mColor.g = scale_color_channel(light->mColor.g, diffuseMultiplier);
            light->mColor.b = scale_color_channel(light->mColor.b, diffuseMultiplier);
        }
    }
}

void release_pipeline() {
    if (g_compositePipeline != nullptr) {
        wgpuRenderPipelineRelease(g_compositePipeline);
        g_compositePipeline = nullptr;
    }
    if (g_compositeLayout != nullptr) {
        wgpuBindGroupLayoutRelease(g_compositeLayout);
        g_compositeLayout = nullptr;
    }
    g_sceneTargetLayout = GFX_RENDER_TARGET_LAYOUT_INIT;
}

bool build_composite_pipeline(const GfxRenderTargetLayout& targetLayout) {
    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl.code = {static_cast<const char*>(g_shaderSource.data), g_shaderSource.size};
    WGPUShaderModuleDescriptor moduleDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    moduleDesc.nextInChain = &wgsl.chain;
    moduleDesc.label = {"phong composite", WGPU_STRLEN};
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(g_deviceInfo.device, &moduleDesc);
    if (module == nullptr) {
        return false;
    }

    WGPUBlendState blendState{
        .color = {.operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_One,
            .dstFactor = WGPUBlendFactor_One},
        .alpha = {.operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_Zero,
            .dstFactor = WGPUBlendFactor_One},
    };
    WGPUColorTargetState colorTargets[GFX_MAX_COLOR_ATTACHMENTS];
    const uint32_t colorTargetCount =
        gfx_init_color_target_states(&targetLayout, colorTargets, &blendState, WGPUColorWriteMask_All);
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module;
    fragment.entryPoint = {"fs_main", WGPU_STRLEN};
    fragment.targetCount = colorTargetCount;
    fragment.targets = colorTargets;
    WGPUDepthStencilState depthStencil = WGPU_DEPTH_STENCIL_STATE_INIT;
    depthStencil.format = targetLayout.depth_stencil_format;
    depthStencil.depthWriteEnabled = WGPUOptionalBool_False;
    depthStencil.depthCompare = WGPUCompareFunction_Always;

    WGPURenderPipelineDescriptor pipelineDesc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    pipelineDesc.label = {"phong composite", WGPU_STRLEN};
    pipelineDesc.vertex.module = module;
    pipelineDesc.vertex.entryPoint = {"vs_main", WGPU_STRLEN};
    pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pipelineDesc.depthStencil = &depthStencil;
    pipelineDesc.multisample.count = targetLayout.sample_count;
    pipelineDesc.fragment = &fragment;
    g_compositePipeline = wgpuDeviceCreateRenderPipeline(g_deviceInfo.device, &pipelineDesc);
    wgpuShaderModuleRelease(module);
    if (g_compositePipeline == nullptr) {
        return false;
    }
    g_compositeLayout = wgpuRenderPipelineGetBindGroupLayout(g_compositePipeline, 0);
    if (g_compositeLayout == nullptr) {
        release_pipeline();
        return false;
    }
    g_sceneTargetLayout = targetLayout;
    return true;
}

bool ensure_pipeline(const GfxRenderTargetLayout& layout) {
    if (g_compositePipeline != nullptr && g_sceneTargetLayout.key == layout.key) {
        return true;
    }
    release_pipeline();
    return build_composite_pipeline(layout);
}

WGPUBindGroup create_bind_group(WGPUDevice device, WGPUBuffer uniformBuffer,
    WGPUBuffer storageBuffer, const DrawPayload& data) {
    if (data.sceneDepth == nullptr || data.sceneNormal == nullptr || g_compositeLayout == nullptr ||
        uniformBuffer == nullptr || storageBuffer == nullptr)
    {
        return nullptr;
    }

    WGPUBindGroupEntry entries[4] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
        WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
    entries[0].binding = 0;
    entries[0].textureView = data.sceneDepth;
    entries[1].binding = 1;
    entries[1].textureView = data.sceneNormal;
    entries[2].binding = 2;
    entries[2].buffer = uniformBuffer;
    entries[2].offset = data.uniform_offset;
    entries[2].size = data.uniform_size;
    entries[3].binding = 3;
    entries[3].buffer = storageBuffer;
    entries[3].offset = data.storage_offset;
    entries[3].size = data.storage_size;
    WGPUBindGroupDescriptor bindGroupDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bindGroupDesc.layout = g_compositeLayout;
    bindGroupDesc.entryCount = 4;
    bindGroupDesc.entries = entries;
    return wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}

void on_draw(
    ModContext*, const GfxDrawContext* ctx, const void* payload, size_t payloadSize, void*) {
    if (payloadSize != sizeof(DrawPayload) || !ensure_pipeline(ctx->layout)) {
        return;
    }
    DrawPayload data;
    std::memcpy(&data, payload, sizeof(data));

    WGPUBindGroup bindGroup =
        create_bind_group(ctx->device, ctx->uniform_buffer, ctx->storage_buffer, data);
    if (bindGroup == nullptr) {
        return;
    }

    wgpuRenderPassEncoderSetPipeline(ctx->pass, g_compositePipeline);
    wgpuRenderPassEncoderSetBindGroup(ctx->pass, 0, bindGroup, 0, nullptr);
    wgpuRenderPassEncoderDraw(ctx->pass, 3, 1, 0, 0);
    wgpuBindGroupRelease(bindGroup);
}

void on_scene_after_opaque(ModContext*, const GfxStageContext* stageCtx, void*) {
    if (!get_bool_option(g_cvarEnabled, false)) {
        return;
    }
    const bool wantSpecular = get_bool_option(g_cvarEnableSpecular, true);
    const bool wantRim = get_bool_option(g_cvarEnableRim, true);
    if (!wantSpecular && !wantRim) {
        return;
    }

    if (stageCtx == nullptr || stageCtx->game_view == nullptr) {
        return;
    }
    CameraInfo camera = CAMERA_INFO_INIT;
    if (svc_camera->get_camera(mod_ctx, stageCtx->game_view, &camera) != MOD_OK) {
        return;
    }

    GfxResolveDesc resolveDesc = GFX_RESOLVE_DESC_INIT;
    resolveDesc.color = false;
    resolveDesc.depth = true;
    resolveDesc.normal = 1;
    GfxResolvedTargets resolved = GFX_RESOLVED_TARGETS_INIT;
    if (svc_gfx->resolve_pass(mod_ctx, &resolveDesc, &resolved) != MOD_OK ||
        resolved.depth == nullptr || resolved.normal == nullptr)
    {
        return;
    }

    GpuLight lights[kMaxLights];
    uint32_t lightCount = wantSpecular ? gather_lights(lights) : 0;
    for (uint32_t i = 0; i < lightCount; ++i) {
        const float x = lights[i].position[0];
        const float y = lights[i].position[1];
        const float z = lights[i].position[2];
        for (int r = 0; r < 3; ++r) {
            lights[i].position[r] = camera.view_from_world[0 * 4 + r] * x +
                                     camera.view_from_world[1 * 4 + r] * y +
                                     camera.view_from_world[2 * 4 + r] * z +
                                     camera.view_from_world[3 * 4 + r];
        }
    }

    PhongUniforms uniforms{};
    std::memcpy(uniforms.view_from_proj, camera.view_from_proj, sizeof(uniforms.view_from_proj));
    const float specularPct =
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarSpecularPct, 20), 0, 100));
    const float rimPct =
        static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarRimPct, 8), 0, 50));
    uniforms.light_count = lightCount;
    uniforms.specular_intensity = specularPct / 100.0f;
    uniforms.rim_intensity = rimPct / 100.0f;
    uniforms.enable_specular = wantSpecular ? 1u : 0u;
    uniforms.enable_rim = wantRim ? 1u : 0u;

    GfxRange uniformRange{0, 0};
    if (svc_gfx->push_uniform(mod_ctx, &uniforms, sizeof(uniforms), &uniformRange) != MOD_OK) {
        return;
    }
    GfxRange storageRange{0, 0};
    const uint32_t storageLightCount = std::max<uint32_t>(lightCount, 1);
    if (lightCount == 0) {
        std::memset(&lights[0], 0, sizeof(GpuLight));
    }
    if (svc_gfx->push_storage(mod_ctx, lights, storageLightCount * sizeof(GpuLight), &storageRange) !=
        MOD_OK)
    {
        return;
    }
    const DrawPayload payload{resolved.depth, resolved.normal, uniformRange.offset,
        uniformRange.size, storageRange.offset, storageRange.size};
    svc_gfx->push_draw(mod_ctx, g_drawType, &payload, sizeof(payload));
}

void add_control(UiElementHandle pane, const UiControlDesc& desc) {
    svc_ui->pane_add_control(mod_ctx, pane, &desc, nullptr);
}

void add_toggle(UiElementHandle pane, const char* label, ConfigVarHandle cvar) {
    UiControlDesc control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_TOGGLE;
    control.label = label;
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = cvar;
    add_control(pane, control);
}

void add_number(UiElementHandle pane, const char* label, ConfigVarHandle cvar, int64_t min,
    int64_t max, int64_t step, const char* suffix) {
    UiControlDesc control = UI_CONTROL_DESC_INIT;
    control.kind = UI_CONTROL_NUMBER;
    control.label = label;
    control.binding = UI_BINDING_CONFIG_VAR;
    control.config_var = cvar;
    control.min = min;
    control.max = max;
    control.step = step;
    control.suffix = suffix;
    add_control(pane, control);
}

ModResult build_panel(ModContext*, UiElementHandle panel, void*, ModError*) {
    add_toggle(panel, "Enabled", g_cvarEnabled);
    add_toggle(panel, "Specular Highlights", g_cvarEnableSpecular);
    add_number(panel, "Specular Intensity", g_cvarSpecularPct, 0, 100, 5, "%");
    add_toggle(panel, "Rim Lighting", g_cvarEnableRim);
    add_number(panel, "Rim Intensity", g_cvarRimPct, 0, 50, 1, "%");
    add_number(panel, "Ambient Multiplier", g_cvarAmbientPct, 0, 300, 5, "%");
    add_number(panel, "Diffuse Multiplier", g_cvarDiffusePct, 0, 300, 5, "%");
    return MOD_OK;
}

ModResult register_bool_option(
    const char* name, bool defaultValue, ConfigVarHandle& outHandle, ModError* error) {
    ConfigVarDesc cvarDesc = CONFIG_VAR_DESC_INIT;
    cvarDesc.name = name;
    cvarDesc.type = CONFIG_VAR_BOOL;
    cvarDesc.default_bool = defaultValue;
    if (svc_config->register_var(mod_ctx, &cvarDesc, &outHandle) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register phong lighting option");
    }
    return MOD_OK;
}

ModResult register_int_option(
    const char* name, int64_t defaultValue, ConfigVarHandle& outHandle, ModError* error) {
    ConfigVarDesc cvarDesc = CONFIG_VAR_DESC_INIT;
    cvarDesc.name = name;
    cvarDesc.type = CONFIG_VAR_INT;
    cvarDesc.default_int = defaultValue;
    if (svc_config->register_var(mod_ctx, &cvarDesc, &outHandle) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register phong lighting option");
    }
    return MOD_OK;
}

}

extern "C" {

MOD_EXPORT ModResult mod_initialize(ModError* error) {
    ModResult result = svc_resource->load(mod_ctx, "phong.wgsl", &g_shaderSource);
    if (result != MOD_OK || g_shaderSource.data == nullptr) {
        return mods::set_error(error, result, "failed to load phong.wgsl");
    }

    result = register_bool_option("effectEnabled", false, g_cvarEnabled, error);
    if (result != MOD_OK) {
        return result;
    }
    result = register_bool_option("enableSpecular", true, g_cvarEnableSpecular, error);
    if (result != MOD_OK) {
        return result;
    }
    result = register_bool_option("enableRim", true, g_cvarEnableRim, error);
    if (result != MOD_OK) {
        return result;
    }
    result = register_int_option("specularIntensity", 20, g_cvarSpecularPct, error);
    if (result != MOD_OK) {
        return result;
    }
    result = register_int_option("rimIntensity", 8, g_cvarRimPct, error);
    if (result != MOD_OK) {
        return result;
    }
    result = register_int_option("ambientMultiplier", 100, g_cvarAmbientPct, error);
    if (result != MOD_OK) {
        return result;
    }
    result = register_int_option("diffuseMultiplier", 100, g_cvarDiffusePct, error);
    if (result != MOD_OK) {
        return result;
    }

    if (svc_gfx->get_device_info(mod_ctx, &g_deviceInfo) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to query device info");
    }

    GfxDrawTypeDesc drawDesc = GFX_DRAW_TYPE_DESC_INIT;
    drawDesc.label = "phong composite";
    drawDesc.draw = on_draw;
    if (svc_gfx->register_draw_type(mod_ctx, &drawDesc, &g_drawType) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register draw type");
    }
    GfxStageHookDesc stageDesc = GFX_STAGE_HOOK_DESC_INIT;
    stageDesc.callback = on_scene_after_opaque;
    if (svc_gfx->register_stage_hook(
            mod_ctx, GFX_STAGE_SCENE_AFTER_OPAQUE, &stageDesc, &g_afterOpaqueHook) != MOD_OK)
    {
        return mods::set_error(error, MOD_ERROR, "failed to register stage hook");
    }
    if (mods::hook::add_post<SetLightNowroomActor>(on_set_light_nowroom_actor_post) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to hook dKy_setLight_nowroom_actor");
    }

    UiModsPanelDesc panelDesc = UI_MODS_PANEL_DESC_INIT;
    panelDesc.build = build_panel;
    svc_ui->register_mods_panel(mod_ctx, &panelDesc);

    return MOD_OK;
}

MOD_EXPORT ModResult mod_update(ModError*) {
    return MOD_OK;
}

MOD_EXPORT ModResult mod_shutdown(ModError*) {
    svc_resource->free(mod_ctx, &g_shaderSource);
    release_pipeline();
    g_cvarEnabled = g_cvarEnableSpecular = g_cvarEnableRim = 0;
    g_cvarSpecularPct = g_cvarRimPct = g_cvarAmbientPct = g_cvarDiffusePct = 0;
    g_drawType = g_afterOpaqueHook = 0;
    return MOD_OK;
}

}
