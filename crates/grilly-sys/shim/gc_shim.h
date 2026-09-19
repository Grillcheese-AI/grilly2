/* gc_shim.h — the whole of what grilly2's core may ask of v1's Vulkan layer.
 *
 * Every function returns GC_OK or GC_ERR. On GC_ERR the message is available
 * from gc_last_error() on the same thread until the next failing call.
 * No C++ exception crosses this boundary.
 *
 * Ownership: the caller owns the context and every buffer handle it creates,
 * and must release handles before destroying the context. Descriptor sets,
 * pipelines and allocations stay on the C++ side and are never exposed.
 *
 * Batching: gc_begin opens a command batch, gc_dispatch / gc_copy / gc_fill /
 * gc_barrier record into it, gc_submit submits it and waits. Upload and download are refused while a
 * batch is open, because they run immediately and would race recorded work.
 * A buffer released while a batch is open is returned to the pool only after
 * that batch has been submitted.
 */
#ifndef GC_SHIM_H
#define GC_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gc_ctx gc_ctx;
typedef uint64_t gc_buf; /* 0 is never a valid handle */

#define GC_OK 0
#define GC_ERR 1

/* Memory class of a buffer. */
#define GC_MEM_DEVICE 0   /* VRAM, fastest for kernels; staged transfers */
#define GC_MEM_UPLOAD 1   /* host-visible, write-combined; CPU writes */
#define GC_MEM_READBACK 2 /* host-visible, cached; CPU reads */

typedef struct gc_caps {
    char device_name[256];
    uint32_t has_cooperative_matrix;
    uint32_t has_float16;
    uint64_t min_storage_buffer_offset_alignment;
    uint32_t max_push_constants_size;
    uint32_t max_compute_workgroup_count[3];
} gc_caps;

/* One storage-buffer binding. range == 0 means "to the end of the buffer". */
typedef struct gc_binding {
    gc_buf buf;
    uint64_t offset;
    uint64_t range;
} gc_binding;

const char* gc_last_error(void);

int32_t gc_create(gc_ctx** out);
void gc_destroy(gc_ctx* ctx);
int32_t gc_caps_get(gc_ctx* ctx, gc_caps* out);

int32_t gc_load_shader(gc_ctx* ctx, const char* name, const char* spv_path);
/* Compile a GLSL compute shader with the shaderc linked into v1 (the same
   settings as the build's glslc: -O, Vulkan 1.2) and register it under
   `name` in v1's PipelineCache. For generated (fused) kernels. */
int32_t gc_compile_shader(gc_ctx* ctx, const char* name, const char* glsl, uint64_t len);
int32_t gc_load_shader_dir(gc_ctx* ctx, const char* dir, uint32_t* loaded);
int32_t gc_has_shader(gc_ctx* ctx, const char* name, uint32_t* out);

int32_t gc_buffer_create(gc_ctx* ctx, uint64_t bytes, uint32_t mem, gc_buf* out);
int32_t gc_buffer_release(gc_ctx* ctx, gc_buf buf);
int32_t gc_buffer_size(gc_ctx* ctx, gc_buf buf, uint64_t* out);

/* Device-memory uploads currently require offset == 0. */
int32_t gc_upload(gc_ctx* ctx, gc_buf dst, uint64_t offset, const void* src,
                  uint64_t bytes);
int32_t gc_download(gc_ctx* ctx, gc_buf src, uint64_t offset, void* dst,
                    uint64_t bytes);

int32_t gc_begin(gc_ctx* ctx);
int32_t gc_dispatch(gc_ctx* ctx, const char* shader, const gc_binding* bindings,
                    uint32_t num_bindings, uint32_t gx, uint32_t gy, uint32_t gz,
                    const void* push, uint32_t push_bytes);
/* Orders all earlier recorded work (dispatch, copy, fill) before all later work. */
int32_t gc_barrier(gc_ctx* ctx);
/* Device-side copy of `bytes` from src+src_offset to dst+dst_offset. */
int32_t gc_copy(gc_ctx* ctx, gc_buf src, uint64_t src_offset, gc_buf dst,
                uint64_t dst_offset, uint64_t bytes);
/* Fill `bytes` at `offset` with the 32-bit pattern `word`; both multiples of 4. */
int32_t gc_fill(gc_ctx* ctx, gc_buf dst, uint64_t offset, uint64_t bytes, uint32_t word);
int32_t gc_submit(gc_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* GC_SHIM_H */
