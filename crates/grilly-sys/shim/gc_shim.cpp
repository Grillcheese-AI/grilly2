// gc_shim.cpp — extern "C" shim over v1's Vulkan primitives.
//
// Uses GrillyDevice, BufferPool, CommandBatch and PipelineCache::getOrCreate
// from grilly_core_lib. Descriptor sets are allocated here, not through
// PipelineCache::allocDescriptorSet, whose cache key ignores binding offsets
// (see docs/boundary.md).

#include "gc_shim.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "grilly/vulkan/vk_buffer_pool.h"
#include "grilly/vulkan/vk_command_batch.h"
#include "grilly/vulkan/vk_device.h"
#include "grilly/vulkan/vk_pipeline_cache.h"

namespace {

thread_local std::string g_last_error;

constexpr uint32_t kSetsPerPool = 1024;
constexpr uint32_t kMaxBindings = 32;

struct Signature {
    uint32_t numBindings;
    uint32_t pushBytes;
};

}  // namespace

struct gc_ctx {
    grilly::GrillyDevice device;
    grilly::BufferPool pool;
    grilly::PipelineCache cache;
    grilly::CommandBatch batch;

    std::mutex mutex;
    VkPhysicalDeviceProperties props{};
    bool recording = false;

    std::unordered_map<gc_buf, grilly::GrillyBuffer> buffers;
    gc_buf nextHandle = 1;
    std::vector<grilly::GrillyBuffer> deferredRelease;

    std::unordered_map<std::string, Signature> signatures;

    // Descriptor pools for the open batch. All are reset after submit;
    // `activePool` indexes the one currently being allocated from.
    std::vector<VkDescriptorPool> descPools;
    size_t activePool = 0;

    gc_ctx() : device(), pool(device), cache(device), batch(device) {
        vkGetPhysicalDeviceProperties(device.physicalDevice(), &props);
    }

    ~gc_ctx() {
        if (batch.isRecording() || batch.isPending()) {
            try {
                batch.submit();
            } catch (...) {
            }
        }
        vkDeviceWaitIdle(device.device());
        for (auto& [h, buf] : buffers) pool.release(buf);
        for (auto& buf : deferredRelease) pool.release(buf);
        for (VkDescriptorPool p : descPools)
            vkDestroyDescriptorPool(device.device(), p, nullptr);
    }

    grilly::GrillyBuffer& lookup(gc_buf h) {
        auto it = buffers.find(h);
        if (it == buffers.end())
            throw std::runtime_error("invalid buffer handle " + std::to_string(h));
        return it->second;
    }

    VkDescriptorPool newDescPool() {
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                  kSetsPerPool * 8};
        VkDescriptorPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.maxSets = kSetsPerPool;
        info.poolSizeCount = 1;
        info.pPoolSizes = &size;
        VkDescriptorPool p = VK_NULL_HANDLE;
        if (vkCreateDescriptorPool(device.device(), &info, nullptr, &p) != VK_SUCCESS)
            throw std::runtime_error("vkCreateDescriptorPool failed");
        descPools.push_back(p);
        return p;
    }

    VkDescriptorSet allocSet(VkDescriptorSetLayout layout) {
        if (descPools.empty()) newDescPool();
        for (;;) {
            VkDescriptorSetAllocateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            info.descriptorPool = descPools[activePool];
            info.descriptorSetCount = 1;
            info.pSetLayouts = &layout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            VkResult r = vkAllocateDescriptorSets(device.device(), &info, &set);
            if (r == VK_SUCCESS) return set;
            if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL)
                throw std::runtime_error("vkAllocateDescriptorSets failed");
            if (++activePool == descPools.size()) newDescPool();
        }
    }

    void resetDescPools() {
        for (VkDescriptorPool p : descPools)
            vkResetDescriptorPool(device.device(), p, 0);
        activePool = 0;
    }
};

namespace {

template <typename F>
int32_t guarded(F&& f) {
    try {
        f();
        return GC_OK;
    } catch (const std::exception& e) {
        g_last_error = e.what();
    } catch (...) {
        g_last_error = "unknown C++ exception";
    }
    return GC_ERR;
}

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

void requireCtx(gc_ctx* ctx) { require(ctx != nullptr, "null gc_ctx"); }

}  // namespace

extern "C" {

const char* gc_last_error(void) { return g_last_error.c_str(); }

int32_t gc_create(gc_ctx** out) {
    return guarded([&] {
        require(out != nullptr, "null out pointer");
        *out = nullptr;
        *out = new gc_ctx();
    });
}

void gc_destroy(gc_ctx* ctx) { delete ctx; }

int32_t gc_caps_get(gc_ctx* ctx, gc_caps* out) {
    return guarded([&] {
        requireCtx(ctx);
        require(out != nullptr, "null out pointer");
        std::memset(out, 0, sizeof(*out));
        const std::string& name = ctx->device.deviceName();
        std::memcpy(out->device_name, name.data(),
                    std::min(name.size(), sizeof(out->device_name) - 1));
        out->has_cooperative_matrix = ctx->device.hasCooperativeMatrix() ? 1 : 0;
        out->has_float16 = ctx->device.hasFloat16() ? 1 : 0;
        const VkPhysicalDeviceLimits& lim = ctx->props.limits;
        out->min_storage_buffer_offset_alignment = lim.minStorageBufferOffsetAlignment;
        out->max_push_constants_size = lim.maxPushConstantsSize;
        for (int i = 0; i < 3; ++i)
            out->max_compute_workgroup_count[i] = lim.maxComputeWorkGroupCount[i];
    });
}

int32_t gc_load_shader(gc_ctx* ctx, const char* name, const char* spv_path) {
    return guarded([&] {
        requireCtx(ctx);
        require(name && spv_path, "null shader name or path");
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->cache.loadSPIRVFile(name, spv_path);
    });
}

int32_t gc_load_shader_dir(gc_ctx* ctx, const char* dir, uint32_t* loaded) {
    return guarded([&] {
        requireCtx(ctx);
        require(dir != nullptr, "null directory");
        namespace fs = std::filesystem;
        fs::path path = fs::u8path(dir);
        require(fs::is_directory(path), std::string("shader directory not found: ") + dir);
        std::lock_guard<std::mutex> lock(ctx->mutex);
        uint32_t count = 0;
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.path().extension() != ".spv") continue;
            ctx->cache.registerSPIRVFile(entry.path().stem().string(),
                                         entry.path().string());
            ++count;
        }
        if (loaded) *loaded = count;
    });
}

int32_t gc_has_shader(gc_ctx* ctx, const char* name, uint32_t* out) {
    return guarded([&] {
        requireCtx(ctx);
        require(name && out, "null argument");
        std::lock_guard<std::mutex> lock(ctx->mutex);
        *out = ctx->cache.hasShader(name) ? 1 : 0;
    });
}

int32_t gc_buffer_create(gc_ctx* ctx, uint64_t bytes, uint32_t mem, gc_buf* out) {
    return guarded([&] {
        requireCtx(ctx);
        require(out != nullptr, "null out pointer");
        require(bytes > 0, "buffer size must be > 0");
        *out = 0;
        std::lock_guard<std::mutex> lock(ctx->mutex);
        grilly::GrillyBuffer buf;
        switch (mem) {
            case GC_MEM_DEVICE: buf = ctx->pool.acquireDeviceLocal(bytes); break;
            case GC_MEM_UPLOAD: buf = ctx->pool.acquire(bytes); break;
            case GC_MEM_READBACK: buf = ctx->pool.acquireReadback(bytes); break;
            default: throw std::runtime_error("unknown memory class " + std::to_string(mem));
        }
        gc_buf h = ctx->nextHandle++;
        ctx->buffers.emplace(h, buf);
        *out = h;
    });
}

int32_t gc_buffer_release(gc_ctx* ctx, gc_buf h) {
    return guarded([&] {
        requireCtx(ctx);
        std::lock_guard<std::mutex> lock(ctx->mutex);
        auto it = ctx->buffers.find(h);
        require(it != ctx->buffers.end(), "invalid buffer handle " + std::to_string(h));
        grilly::GrillyBuffer buf = it->second;
        ctx->buffers.erase(it);
        if (ctx->recording)
            ctx->deferredRelease.push_back(buf);
        else
            ctx->pool.release(buf);
    });
}

int32_t gc_buffer_size(gc_ctx* ctx, gc_buf h, uint64_t* out) {
    return guarded([&] {
        requireCtx(ctx);
        require(out != nullptr, "null out pointer");
        std::lock_guard<std::mutex> lock(ctx->mutex);
        *out = ctx->lookup(h).size;
    });
}

int32_t gc_upload(gc_ctx* ctx, gc_buf dst, uint64_t offset, const void* src,
                  uint64_t bytes) {
    return guarded([&] {
        requireCtx(ctx);
        require(src != nullptr || bytes == 0, "null source pointer");
        std::lock_guard<std::mutex> lock(ctx->mutex);
        require(!ctx->recording, "gc_upload called while a batch is open");
        grilly::GrillyBuffer& buf = ctx->lookup(dst);
        require(offset <= buf.size && bytes <= buf.size - offset,
                "upload out of bounds");
        if (bytes == 0) return;
        if (buf.mappedPtr) {
            std::memcpy(static_cast<char*>(buf.mappedPtr) + offset, src, bytes);
            vmaFlushAllocation(ctx->pool.allocator(), buf.allocation, offset, bytes);
        } else {
            require(offset == 0, "upload at a nonzero offset into device memory "
                                 "is not supported yet");
            ctx->pool.uploadStaged(buf, src, bytes);
        }
    });
}

int32_t gc_download(gc_ctx* ctx, gc_buf srcBuf, uint64_t offset, void* dst,
                    uint64_t bytes) {
    return guarded([&] {
        requireCtx(ctx);
        require(dst != nullptr || bytes == 0, "null destination pointer");
        std::lock_guard<std::mutex> lock(ctx->mutex);
        require(!ctx->recording, "gc_download called while a batch is open");
        grilly::GrillyBuffer& buf = ctx->lookup(srcBuf);
        require(offset <= buf.size && bytes <= buf.size - offset,
                "download out of bounds");
        if (bytes == 0) return;
        if (buf.mappedPtr)
            ctx->pool.download(buf, static_cast<float*>(dst), bytes, offset);
        else
            ctx->pool.downloadStaged(buf, dst, bytes, offset);
    });
}

int32_t gc_begin(gc_ctx* ctx) {
    return guarded([&] {
        requireCtx(ctx);
        std::lock_guard<std::mutex> lock(ctx->mutex);
        require(!ctx->recording, "gc_begin called while a batch is already open");
        ctx->batch.begin();
        ctx->recording = true;
    });
}

int32_t gc_dispatch(gc_ctx* ctx, const char* shader, const gc_binding* bindings,
                    uint32_t num_bindings, uint32_t gx, uint32_t gy, uint32_t gz,
                    const void* push, uint32_t push_bytes) {
    return guarded([&] {
        requireCtx(ctx);
        require(shader != nullptr, "null shader name");
        require(bindings != nullptr || num_bindings == 0, "null bindings");
        require(num_bindings <= kMaxBindings, "too many bindings");
        require(push != nullptr || push_bytes == 0, "null push-constant data");
        std::lock_guard<std::mutex> lock(ctx->mutex);
        require(ctx->recording, "gc_dispatch called without gc_begin");

        const VkPhysicalDeviceLimits& lim = ctx->props.limits;
        require(push_bytes <= lim.maxPushConstantsSize, "push constants too large");
        require(gx > 0 && gy > 0 && gz > 0, "workgroup counts must be > 0");
        require(gx <= lim.maxComputeWorkGroupCount[0] &&
                    gy <= lim.maxComputeWorkGroupCount[1] &&
                    gz <= lim.maxComputeWorkGroupCount[2],
                "workgroup count exceeds device limit");

        // Pipelines are cached by name alone, so a shader must always be
        // dispatched with the layout it was first created with.
        auto sig = ctx->signatures.find(shader);
        if (sig != ctx->signatures.end()) {
            require(sig->second.numBindings == num_bindings &&
                        sig->second.pushBytes == push_bytes,
                    std::string("shader '") + shader +
                        "' dispatched with a different binding count or push "
                        "size than its first use");
        }

        const uint64_t align = lim.minStorageBufferOffsetAlignment;
        VkDescriptorBufferInfo infos[kMaxBindings];
        for (uint32_t i = 0; i < num_bindings; ++i) {
            const gc_binding& b = bindings[i];
            grilly::GrillyBuffer& buf = ctx->lookup(b.buf);
            require(b.offset < buf.size, "binding offset out of bounds");
            require(align == 0 || b.offset % align == 0,
                    "binding offset not aligned to minStorageBufferOffsetAlignment (" +
                        std::to_string(align) + ")");
            uint64_t range = b.range ? b.range : buf.size - b.offset;
            require(range <= buf.size - b.offset, "binding range out of bounds");
            infos[i] = {buf.handle, b.offset, range};
        }

        grilly::PipelineEntry pipe =
            ctx->cache.getOrCreate(shader, num_bindings, push_bytes);
        ctx->signatures.emplace(shader, Signature{num_bindings, push_bytes});
        VkDescriptorSet set = ctx->allocSet(pipe.descLayout);

        VkWriteDescriptorSet writes[kMaxBindings];
        for (uint32_t i = 0; i < num_bindings; ++i) {
            writes[i] = {};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(ctx->device.device(), num_bindings, writes, 0, nullptr);

        ctx->batch.dispatch(pipe.pipeline, pipe.layout, set, gx, gy, gz, push,
                            push_bytes);
    });
}

int32_t gc_barrier(gc_ctx* ctx) {
    return guarded([&] {
        requireCtx(ctx);
        std::lock_guard<std::mutex> lock(ctx->mutex);
        require(ctx->recording, "gc_barrier called without gc_begin");
        // Unlike CommandBatch::barrier, also orders write-after-write.
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(ctx->batch.cmdBuffer(),
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    });
}

int32_t gc_submit(gc_ctx* ctx) {
    return guarded([&] {
        requireCtx(ctx);
        std::lock_guard<std::mutex> lock(ctx->mutex);
        require(ctx->recording, "gc_submit called without gc_begin");
        ctx->recording = false;
        ctx->batch.submit();  // submit + wait
        ctx->resetDescPools();
        for (auto& buf : ctx->deferredRelease) ctx->pool.release(buf);
        ctx->deferredRelease.clear();
    });
}

}  // extern "C"
