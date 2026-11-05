// SPDX-License-Identifier: GPL-3.0-or-later

// EDEN_NX2_DMA_TRACE=1: diagnostics for draws whose arguments come from native MME DMA reads
// (macro reads of GPU memory). Logs each DMA source the first time it is read and every time its
// words change, links zero-count task/mesh draws to the last DMA read, and prints a per-shader
// summary of zero-count mesh draws periodically. Off by default; no cost when unset.

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <mutex>
#include <span>
#include <tuple>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::Nx2DmaTrace {

inline bool Enabled() {
    static const bool enabled = std::getenv("EDEN_NX2_DMA_TRACE") != nullptr;
    return enabled;
}

struct State {
    std::mutex mutex;
    std::map<u64, std::array<u32, 8>> last_words;
    u64 last_source{};
    u32 last_method{};
    std::array<u32, 8> last_read{};
    bool last_dirty{};
    u32 change_logs{};
    u32 zero_logs{};
    u32 write_logs{};
    u64 mesh_total{};
    u64 mesh_zero{};
    std::map<std::tuple<u64, u64>, std::array<u64, 2>> per_shader; // (vs, ps) -> {zero, total}
};

inline State& Get() {
    static State state;
    return state;
}

/// Called by the macro engine after an MME DMA read.
inline void NoteDmaRead(u32 method, u64 source, std::span<const u32> words, bool was_dirty) {
    if (!Enabled()) return;
    auto& s = Get();
    std::scoped_lock lock{s.mutex};
    std::array<u32, 8> head{};
    std::copy_n(words.begin(), std::min<size_t>(words.size(), head.size()), head.begin());
    s.last_source = source;
    s.last_method = method;
    s.last_read = head;
    s.last_dirty = was_dirty;
    const auto [it, inserted] = s.last_words.try_emplace(source, head);
    if ((inserted || it->second != head) && s.change_logs < 3000) {
        ++s.change_logs;
        LOG_INFO(HW_GPU, "DMA trace {} source={:#x} start={:#x} dirty={} words={:08x}",
                 inserted ? "new" : "changed", source, method, was_dirty, fmt::join(head, ","));
        it->second = head;
    }
}

/// Called by the macro engine for each method write made after the call's first DMA read.
inline void NoteMacroWrite(u32 start, u32 address, u32 value, const char* name) {
    if (!Enabled()) return;
    auto& s = Get();
    std::scoped_lock lock{s.mutex};
    if (s.write_logs < 6000) {
        ++s.write_logs;
        LOG_INFO(HW_GPU, "DMA trace write start={:#x} method={:#x} {} value={:#x}", start, address,
                 name, value);
    }
}

/// Called by the rasterizer for every direct task/mesh draw.
inline void NoteMeshDraw(u32 groups, u64 vs, u64 ps, u64 frame, u32 raw_count, u32 topology,
                         bool indexed) {
    if (!Enabled()) return;
    auto& s = Get();
    std::scoped_lock lock{s.mutex};
    ++s.mesh_total;
    auto& counts = s.per_shader[{vs, ps}];
    ++counts[1];
    if (groups == 0) {
        ++s.mesh_zero;
        ++counts[0];
        if (s.zero_logs < 300) {
            ++s.zero_logs;
            LOG_INFO(HW_GPU,
                     "DMA trace zero mesh draw frame={} vs={:#x} ps={:#x} raw_count={} "
                     "topology={:#x} indexed={} last_dma source={:#x} start={:#x} dirty={} "
                     "words={:08x}",
                     frame, vs, ps, raw_count, topology, indexed, s.last_source, s.last_method,
                     s.last_dirty, fmt::join(s.last_read, ","));
        }
    }
    if (s.mesh_total % 20000 == 0) {
        LOG_INFO(HW_GPU, "DMA trace summary: mesh draws {} zero {} distinct sources {}", s.mesh_total,
                 s.mesh_zero, s.last_words.size());
        for (const auto& [key, value] : s.per_shader) {
            if (value[0] != 0) {
                LOG_INFO(HW_GPU, "DMA trace summary:   vs={:#x} ps={:#x} zero {} of {}",
                         std::get<0>(key), std::get<1>(key), value[0], value[1]);
            }
        }
    }
}

} // namespace VideoCommon::Nx2DmaTrace
