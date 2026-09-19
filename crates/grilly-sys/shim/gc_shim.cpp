// gc_shim.cpp — extern "C" shim over v1's Vulkan primitives.
//
// Uses GrillyDevice, BufferPool, CommandBatch and PipelineCache::getOrCreate
// from grilly_core_lib. Descriptor sets are allocated here, not through
// PipelineCache::allocDescriptorSet, whose cache key ignores binding offsets
// (see docs/boundary.md).

#include "gc_shim.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <shaderc/shaderc.h>

#include "grilly/vulkan/vk_buffer_pool.h"
#include "grilly/vulkan/vk_command_batch.h"
#include "grilly/vulkan/vk_device.h"
#include "grilly/vulkan/vk_pipeline_cache.h"

namespace {

thread_local std::string g_last_error;

constexpr uint32_t kSetsPerPool = 1024;
// GC_MEM_DEVICE_PERSISTENT buffers this large get a device memory allocation of their own
constexpr size_t kDedicatedMin = size_t(1) << 20;
// Host <-> device-local transfers are staged this many bytes at a time
constexpr uint64_t kStagingBytes = uint64_t(64) << 20;
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

    // Device-local buffers freed while a batch is recording, by bucket
    // size, reusable by later commands in the same batch. Every recorded
    // command sits behind a full barrier, so a buffer whose last use is
    // recorded can be rewritten by a later command. They are NOT handed to
    // v1's BufferPool mid-batch: its release() destroys a buffer when a
    // bucket is full, which would free one the unsubmitted command buffer
    // still uses. Without this, a batch's footprint was the sum of all its
    // intermediates (CLAUDE.md §11: 20 GB and 12x slower at S=4096).
    std::unordered_map<size_t, std::vector<grilly::GrillyBuffer>> batchFree;
    uint64_t batchReuses = 0;

    // v1 BufferPool's bucket rule (private there): powers of two from 256 B
    // to 256 MB, the exact size above. A mismatch would only miss a reuse.
    static size_t bucketFor(size_t bytes) {
        size_t bucket = size_t(1) << 8;
        while (bucket < bytes && bucket < (size_t(1) << 28)) bucket <<= 1;
        return bucket < bytes ? bytes : bucket;
    }

    grilly::GrillyBuffer acquireDeviceLocal(size_t bytes) {
        auto it = batchFree.find(bucketFor(bytes));
        if (recording && it != batchFree.end() && !it->second.empty() && it->second.back().bucketSize >= bytes) {
            grilly::GrillyBuffer buf = it->second.back();
            it->second.pop_back();
            buf.size = bytes;
            ++batchReuses;
            return buf;
        }
        return pool.acquireDeviceLocal(bytes);
    }

    // After a batch has finished: everything freed during it goes back to v1,
    // or is destroyed if persistent.
    void releaseAfterBatch() {
        for (auto& buf : deferredRelease) pool.release(buf);
        deferredRelease.clear();
        for (auto& buf : deferredDestroy) destroy(buf);
        deferredDestroy.clear();
        for (auto& [bucket, bufs] : batchFree) {
            for (auto& buf : bufs) pool.release(buf);
        }
        batchFree.clear();
    }

    // GC_MEM_DEVICE_PERSISTENT buffers: allocated at their exact size and
    // destroyed when released, never handed to v1's pool. The pool rounds up
    // to a power of two and keeps up to 32 released buffers per size, which
    // held Qwen2.5-0.5B's 1.84 GiB of float32 weights in 3.00 GiB of VRAM.
    std::unordered_set<gc_buf> persistent;
    uint64_t persistentBytes = 0;
    // released while a batch was recording: destroyed once it has run
    std::vector<grilly::GrillyBuffer> deferredDestroy;

    grilly::GrillyBuffer allocatePersistent(size_t bytes) {
        // as v1's BufferPool::acquireDeviceLocal, at the exact size
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        alloc.priority = 1.0f;
        // A large buffer gets device memory of its own: freeing it returns
        // exactly that memory, where a sub-allocation would share a block
        // with pooled buffers that keep the block alive. Small ones (biases,
        // norm weights) share blocks, to stay far below the driver's
        // allocation-count limit (4096 on Windows).
        if (bytes >= kDedicatedMin) alloc.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        grilly::GrillyBuffer buf{};
        buf.size = bytes;
        buf.bucketSize = bytes;
        buf.deviceLocal = true;
        VkResult r = vmaCreateBuffer(pool.allocator(), &info, &alloc, &buf.handle, &buf.allocation, &buf.info);
        if (r != VK_SUCCESS)
            throw std::runtime_error("allocating " + std::to_string(bytes) +
                                     " bytes of persistent device memory failed (VkResult " +
                                     std::to_string(static_cast<int>(r)) + ")");
        return buf;
    }

    void destroy(grilly::GrillyBuffer& buf) { vmaDestroyBuffer(pool.allocator(), buf.handle, buf.allocation); }

    // Transfers between the host and device-local memory go through two
    // staging buffers of kStagingBytes owned here, a chunk at a time. v1's
    // BufferPool::uploadStaged / downloadStaged instead keep a per-thread
    // staging buffer as large as the largest transfer ever made, never freed
    // (0.51 GiB of system RAM after one upload of Qwen2.5-0.5B's embedding),
    // and upload only at offset 0.
    struct Staging {
        VkBuffer handle = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        void* mapped = nullptr;
    };
    Staging upStage, downStage;

    Staging& staging(Staging& s, bool readback) {
        if (s.handle) return s;
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = kStagingBytes;
        info.usage = readback ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo alloc{};
        alloc.usage = VMA_MEMORY_USAGE_AUTO;
        // dedicated: exactly kStagingBytes of system memory each, sharing no block
        alloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT |
                      (readback ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                                : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        VmaAllocationInfo got{};
        VkResult r = vmaCreateBuffer(pool.allocator(), &info, &alloc, &s.handle, &s.allocation, &got);
        if (r != VK_SUCCESS)
            throw std::runtime_error("allocating a staging buffer failed (VkResult " +
                                     std::to_string(static_cast<int>(r)) + ")");
        s.mapped = got.pMappedData;
        return s;
    }

    // Record one copy and a barrier making it visible to `dstStage` /
    // `dstAccess`, run it and wait. No batch may be open.
    void copyNow(VkBuffer from, uint64_t fromOffset, VkBuffer to, uint64_t toOffset, uint64_t bytes,
                 VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
        batch.begin();
        VkBufferCopy region{fromOffset, toOffset, bytes};
        vkCmdCopyBuffer(batch.cmdBuffer(), from, to, 1, &region);
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(batch.cmdBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT, dstStage, 0, 1, &mb, 0, nullptr, 0,
                             nullptr);
        batch.submitDeferred();
        VkResult r = vkQueueWaitIdle(device.computeQueue());  // no timeout, as in gc_submit
        if (r != VK_SUCCESS)
            throw std::runtime_error("vkQueueWaitIdle failed (VkResult=" + std::to_string(r) + ")");
        batch.waitForCompletion();
    }

    void uploadDevice(const grilly::GrillyBuffer& dst, uint64_t offset, const char* src, uint64_t bytes) {
        Staging& s = staging(upStage, false);
        for (uint64_t done = 0; done < bytes;) {
            uint64_t n = std::min(bytes - done, kStagingBytes);
            std::memcpy(s.mapped, src + done, n);
            vmaFlushAllocation(pool.allocator(), s.allocation, 0, n);
            copyNow(s.handle, 0, dst.handle, offset + done, n,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                        VK_ACCESS_TRANSFER_WRITE_BIT);
            done += n;
        }
    }

    void downloadDevice(const grilly::GrillyBuffer& src, uint64_t offset, char* out, uint64_t bytes) {
        Staging& s = staging(downStage, true);
        for (uint64_t done = 0; done < bytes;) {
            uint64_t n = std::min(bytes - done, kStagingBytes);
            copyNow(src.handle, offset + done, s.handle, 0, n, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
            vmaInvalidateAllocation(pool.allocator(), s.allocation, 0, n);
            std::memcpy(out + done, s.mapped, n);
            done += n;
        }
    }

    std::unordered_map<std::string, Signature> signatures;

    // gc_submit_stats counters
    uint64_t submits = 0, dispatches = 0;
    uint64_t submitNs = 0, waitNs = 0, waitNsMax = 0, waitNsLast = 0;

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
        for (auto& [h, buf] : buffers) {
            if (persistent.count(h))
                destroy(buf);
            else
                pool.release(buf);
        }
        for (Staging* s : {&upStage, &downStage})
            if (s->handle) vmaDestroyBuffer(pool.allocator(), s->handle, s->allocation);
        releaseAfterBatch();
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

// v1 logs device bring-up to stdout. A library should not, so the log is
// discarded unless GRILLY_VERBOSE is set; warnings on stderr are untouched.
// std::cout's buffer is process-global, so creations are serialised: two
// overlapping swaps would restore each other's (destroyed) sinks.
std::mutex g_create_mutex;

struct QuietStdout {
    std::ostringstream sink;
    std::streambuf* saved = nullptr;
    QuietStdout() {
        const char* v = std::getenv("GRILLY_VERBOSE");
        if (!v || !*v || std::string(v) == "0") saved = std::cout.rdbuf(sink.rdbuf());
    }
    ~QuietStdout() {
        if (saved) std::cout.rdbuf(saved);
    }
};

int32_t gc_create(gc_ctx** out) {
    return guarded([&] {
        require(out != nullptr, "null out pointer");
        *out = nullptr;
        std::lock_guard<std::mutex> lock(g_create_mutex);
        QuietStdout quiet;
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

int32_t gc_compile_shader(gc_ctx* ctx, const char* name, const char* glsl, uint64_t len) {
    return guarded([&] {
        requireCtx(ctx);
        require(name && glsl, "null shader name or source");
        std::unique_ptr<shaderc_compiler, void (*)(shaderc_compiler_t)> compiler(
            shaderc_compiler_initialize(), shaderc_compiler_release);
        std::unique_ptr<shaderc_compile_options, void (*)(shaderc_compile_options_t)> options(
            shaderc_compile_options_initialize(), shaderc_compile_options_release);
        require(compiler && options, "shaderc failed to initialize");
        shaderc_compile_options_set_optimization_level(options.get(), shaderc_optimization_level_performance);
        shaderc_compile_options_set_target_env(options.get(), shaderc_target_env_vulkan,
                                               shaderc_env_version_vulkan_1_2);
        std::unique_ptr<shaderc_compilation_result, void (*)(shaderc_compilation_result_t)> result(
            shaderc_compile_into_spv(compiler.get(), glsl, static_cast<size_t>(len), shaderc_compute_shader,
                                     name, "main", options.get()),
            shaderc_result_release);
        require(result != nullptr, "shaderc returned no result");
        if (shaderc_result_get_compilation_status(result.get()) != shaderc_compilation_status_success) {
            throw std::runtime_error(std::string("compiling ") + name + ": " +
                                     shaderc_result_get_error_message(result.get()));
        }
        const auto* bytes = reinterpret_cast<const uint8_t*>(shaderc_result_get_bytes(result.get()));
        std::vector<uint8_t> code(bytes, bytes + shaderc_result_get_length(result.get()));
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->cache.loadSPIRV(name, code);
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
            case GC_MEM_DEVICE: buf = ctx->acquireDeviceLocal(bytes); break;
            case GC_MEM_UPLOAD: buf = ctx->pool.acquire(bytes); break;
            case GC_MEM_READBACK: buf = ctx->pool.acquireReadback(bytes); break;
            case GC_MEM_DEVICE_PERSISTENT: buf = ctx->allocatePersistent(bytes); break;
            default: throw std::runtime_error("unknown memory class " + std::to_string(mem));
        }
        gc_buf h = ctx->nextHandle++;
        ctx->buffers.emplace(h, buf);
        if (mem == GC_MEM_DEVICE_PERSISTENT) {
            ctx->persistent.insert(h);
            ctx->persistentBytes += bytes;
        }
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
        if (ctx->persistent.erase(h)) {
            ctx->persistentBytes -= buf.size;
            if (!ctx->recording)
                ctx->destroy(buf);
            else
                ctx->deferredDestroy.push_back(buf);  // recorded commands may still use it
            return;
        }
        if (!ctx->recording)
            ctx->pool.release(buf);
        else if (buf.deviceLocal)
            ctx->batchFree[buf.bucketSize].push_back(buf);  // reusable within this batch
        else
            ctx->deferredRelease.push_back(buf);  // host-visible: only after the batch
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
            ctx->uploadDevice(buf, offset, static_cast<const char*>(src), bytes);
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
            ctx->downloadDevice(buf, offset, static_cast<char*>(dst), bytes);
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
        ctx->dispatches += 1;
    });
}

int32_t gc_barrier(gc_ctx* ctx) {
    return guarded([&] {
        requireCtx(ctx);
        std::lock_guard<std::mutex> lock(ctx->mutex);
        require(ctx->recording, "gc_barrier called without gc_begin");
        // Orders every earlier dispatch, copy and fill before every later
        // one, including write-after-write (CommandBatch::barrier does not).
        const VkPipelineStageFlags stages =
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(ctx->batch.cmdBuffer(), stages, stages, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    });
}

int32_t gc_copy(gc_ctx* ctx, gc_buf src, uint64_t src_offset, gc_buf dst,
                uint64_t dst_offset, uint64_t bytes) {
    return guarded([&] {
        requireCtx(ctx);
        std::lock_guard<std::mutex> lock(ctx->mutex);
        require(ctx->recording, "gc_copy called without gc_begin");
        grilly::GrillyBuffer& s = ctx->lookup(src);
        grilly::GrillyBuffer& d = ctx->lookup(dst);
        require(src_offset <= s.size && bytes <= s.size - src_offset, "copy source out of bounds");
        require(dst_offset <= d.size && bytes <= d.size - dst_offset, "copy destination out of bounds");
        if (bytes == 0) return;
        VkBufferCopy region{src_offset, dst_offset, bytes};
        vkCmdCopyBuffer(ctx->batch.cmdBuffer(), s.handle, d.handle, 1, &region);
    });
}

int32_t gc_fill(gc_ctx* ctx, gc_buf dst, uint64_t offset, uint64_t bytes, uint32_t word) {
    return guarded([&] {
        requireCtx(ctx);
        std::lock_guard<std::mutex> lock(ctx->mutex);
        require(ctx->recording, "gc_fill called without gc_begin");
        grilly::GrillyBuffer& d = ctx->lookup(dst);
        require(offset % 4 == 0 && bytes % 4 == 0, "fill offset and size must be multiples of 4");
        require(offset <= d.size && bytes <= d.size - offset, "fill out of bounds");
        if (bytes == 0) return;
        vkCmdFillBuffer(ctx->batch.cmdBuffer(), d.handle, offset, bytes, word);
    });
}

int32_t gc_submit_stats_get(gc_ctx* ctx, gc_submit_stats* out) {
    return guarded([&] {
        requireCtx(ctx);
        require(out != nullptr, "null stats");
        std::lock_guard<std::mutex> lock(ctx->mutex);
        auto pool = ctx->pool.stats();
        *out = gc_submit_stats{ctx->submits, ctx->dispatches, ctx->submitNs, ctx->waitNs, ctx->waitNsMax,
                               ctx->waitNsLast, static_cast<uint64_t>(ctx->descPools.size()),
                               pool.allocations, pool.totalAcquired - pool.totalReleased, ctx->batchReuses,
                               static_cast<uint64_t>(ctx->persistent.size()), ctx->persistentBytes};
    });
}

int32_t gc_submit(gc_ctx* ctx) {
    return guarded([&] {
        requireCtx(ctx);
        std::lock_guard<std::mutex> lock(ctx->mutex);
        require(ctx->recording, "gc_submit called without gc_begin");
        ctx->recording = false;
        // v1's CommandBatch::submit() is submitDeferred() + waitForCompletion(),
        // and waitForCompletion() waits on the fence with a hard-coded 2 s
        // timeout and throws when it expires — even though the GPU is still
        // working and finishes fine. v1 cannot be changed, so submit in two
        // steps instead: hand the work to the queue, then block on the queue
        // itself, which has no timeout. vkQueueWaitIdle is safe here without
        // extra synchronization because the compute queue is only ever
        // touched while ctx->mutex is held (the lock above).
        using clock = std::chrono::steady_clock;
        auto ns = [](clock::duration d) {
            return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
        };
        auto t0 = clock::now();
        ctx->batch.submitDeferred();
        auto t1 = clock::now();
        VkResult wait = vkQueueWaitIdle(ctx->device.computeQueue());
        auto t2 = clock::now();
        ctx->submits += 1;
        ctx->submitNs += ns(t1 - t0);
        ctx->waitNsLast = ns(t2 - t1);
        ctx->waitNs += ctx->waitNsLast;
        ctx->waitNsMax = std::max(ctx->waitNsMax, ctx->waitNsLast);
        if (wait != VK_SUCCESS) {
            throw std::runtime_error("vkQueueWaitIdle failed (VkResult=" +
                                     std::to_string(wait) + ")");
        }
        // The GPU is idle now, so this just observes the already-signalled
        // fence and clears v1's pending flag.
        ctx->batch.waitForCompletion();
        // Only past this point is it safe to reclaim descriptor sets and
        // buffers: v1 once freed descriptor sets that an in-flight command
        // buffer was still using and got garbage/NaN gradients back (v1
        // commit 49a1a89). Keep the reset and releases after the wait above.
        ctx->resetDescPools();
        ctx->releaseAfterBatch();
    });
}

}  // extern "C"
