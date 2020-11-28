#if GL_core_profile
// Compiling as GLSL
#else
// Compiling as a C header file
typedef struct {
    float x, y, z;
} vec3;
typedef struct {
    float x, y, z, w;
} vec4;
#endif

#define FP_PREVIEW_UNIFORMS \
    vec3 iResolution; \
    float iTime; \
    vec4 iMouse; \

struct fpPreviewUniforms {
    FP_PREVIEW_UNIFORMS
};
