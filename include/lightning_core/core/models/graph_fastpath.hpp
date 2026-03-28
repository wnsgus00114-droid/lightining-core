#pragma once

#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "lightning_core/core/model_customization.hpp"
#include "lightning_core/core/ops.hpp"

namespace lightning_core::models {

struct GraphPolicyCacheKey {
  int family = 0;
  int mode = 0;
  std::size_t num_nodes = 0;
  std::size_t num_edges = 0;
  std::size_t feature_dim = 0;

  bool operator==(const GraphPolicyCacheKey& other) const {
    return family == other.family && mode == other.mode && num_nodes == other.num_nodes &&
           num_edges == other.num_edges && feature_dim == other.feature_dim;
  }
};

struct GraphPolicyCacheKeyHash {
  std::size_t operator()(const GraphPolicyCacheKey& k) const {
    std::size_t h = static_cast<std::size_t>(k.family);
    h = h * 1315423911u + static_cast<std::size_t>(k.mode);
    h = h * 1315423911u + k.num_nodes;
    h = h * 1315423911u + k.num_edges;
    h = h * 1315423911u + k.feature_dim;
    return h;
  }
};

struct GraphSparseFriendlyPolicy {
  // Split edge list processing into chunks to reduce transfer bursts.
  std::size_t edge_chunk_size = 4096;

  // Preferred node mini-batch size for repeated message passing loops.
  std::size_t node_batch_size = 512;

  // If true, use resident IO loop policies for repeated aggregations.
  bool resident_aggregation = true;

  // Route tiny update vectors to CPU fast path when beneficial.
  std::size_t tiny_update_cpu_crossover_n = (2u << 20);
};

struct GraphPolicyCacheState {
  bool loaded = false;
  bool dirty = false;
  std::string path;
  std::unordered_map<GraphPolicyCacheKey, GraphSparseFriendlyPolicy, GraphPolicyCacheKeyHash> table;
  std::mutex mu;
};

inline std::string resolveGraphPolicyCachePath() {
  if (const char* raw = std::getenv("CJ_GRAPH_POLICY_CACHE_FILE")) {
    if (raw[0] != '\0') {
      return std::string(raw);
    }
  }
  return std::string(".lightning_core_graph_policy_cache.csv");
}

inline GraphPolicyCacheState& graphPolicyCacheState() {
  static GraphPolicyCacheState st;
  return st;
}

inline void loadGraphPolicyCacheIfNeeded() {
  GraphPolicyCacheState& st = graphPolicyCacheState();
  std::lock_guard<std::mutex> lock(st.mu);
  if (st.loaded) {
    return;
  }
  st.path = resolveGraphPolicyCachePath();
  st.loaded = true;

  std::ifstream in(st.path);
  if (!in.is_open()) {
    return;
  }

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::stringstream ss(line);
    std::string tok;

    GraphPolicyCacheKey key;
    GraphSparseFriendlyPolicy val;

    if (!std::getline(ss, tok, ',')) continue;
    key.family = std::atoi(tok.c_str());
    if (!std::getline(ss, tok, ',')) continue;
    key.mode = std::atoi(tok.c_str());
    if (!std::getline(ss, tok, ',')) continue;
    key.num_nodes = static_cast<std::size_t>(std::strtoull(tok.c_str(), nullptr, 10));
    if (!std::getline(ss, tok, ',')) continue;
    key.num_edges = static_cast<std::size_t>(std::strtoull(tok.c_str(), nullptr, 10));
    if (!std::getline(ss, tok, ',')) continue;
    key.feature_dim = static_cast<std::size_t>(std::strtoull(tok.c_str(), nullptr, 10));

    if (!std::getline(ss, tok, ',')) continue;
    val.edge_chunk_size = static_cast<std::size_t>(std::strtoull(tok.c_str(), nullptr, 10));
    if (!std::getline(ss, tok, ',')) continue;
    val.node_batch_size = static_cast<std::size_t>(std::strtoull(tok.c_str(), nullptr, 10));
    if (!std::getline(ss, tok, ',')) continue;
    val.resident_aggregation = (std::atoi(tok.c_str()) != 0);
    if (!std::getline(ss, tok, ',')) continue;
    val.tiny_update_cpu_crossover_n = static_cast<std::size_t>(std::strtoull(tok.c_str(), nullptr, 10));

    st.table[key] = val;
  }
}

inline void saveGraphPolicyCacheIfDirty() {
  GraphPolicyCacheState& st = graphPolicyCacheState();
  std::lock_guard<std::mutex> lock(st.mu);
  if (!st.loaded || !st.dirty || st.path.empty()) {
    return;
  }

  std::ofstream out(st.path, std::ios::trunc);
  if (!out.is_open()) {
    return;
  }
  out << "#family,mode,num_nodes,num_edges,feature_dim,edge_chunk_size,node_batch_size,resident_aggregation,tiny_update_cpu_crossover_n\n";
  for (const auto& it : st.table) {
    const GraphPolicyCacheKey& k = it.first;
    const GraphSparseFriendlyPolicy& v = it.second;
    out << k.family << ',' << k.mode << ',' << k.num_nodes << ',' << k.num_edges << ',' << k.feature_dim << ','
        << v.edge_chunk_size << ',' << v.node_batch_size << ',' << (v.resident_aggregation ? 1 : 0) << ','
        << v.tiny_update_cpu_crossover_n << '\n';
  }
  st.dirty = false;
}

inline GraphSparseFriendlyPolicy makeGraphSparseFriendlyPolicy(
    ModelFamily family,
    ExecutionMode mode,
    std::size_t num_nodes,
    std::size_t num_edges,
    std::size_t feature_dim) {
  (void)feature_dim;

  ModelCustomization c = makeAggressiveCustomization(family, mode, 0, 0);

  GraphSparseFriendlyPolicy p;
  p.resident_aggregation = c.resident_io;
  p.tiny_update_cpu_crossover_n = c.vector_oneshot_crossover_n;

  if (family == ModelFamily::kGat) {
    p.edge_chunk_size = (num_edges >= (1u << 20)) ? 16384 : 8192;
    p.node_batch_size = (num_nodes >= (1u << 18)) ? 2048 : 1024;
  } else {
    p.edge_chunk_size = (num_edges >= (1u << 20)) ? 8192 : 4096;
    p.node_batch_size = (num_nodes >= (1u << 18)) ? 1024 : 512;
  }

  return p;
}

inline GraphSparseFriendlyPolicy makeGraphSparseFriendlyPolicyCached(
    ModelFamily family,
    ExecutionMode mode,
    std::size_t num_nodes,
    std::size_t num_edges,
    std::size_t feature_dim) {
  loadGraphPolicyCacheIfNeeded();

  GraphPolicyCacheKey key;
  key.family = static_cast<int>(family);
  key.mode = static_cast<int>(mode);
  key.num_nodes = num_nodes;
  key.num_edges = num_edges;
  key.feature_dim = feature_dim;

  GraphPolicyCacheState& st = graphPolicyCacheState();
  {
    std::lock_guard<std::mutex> lock(st.mu);
    auto it = st.table.find(key);
    if (it != st.table.end()) {
      return it->second;
    }
  }

  GraphSparseFriendlyPolicy p = makeGraphSparseFriendlyPolicy(family, mode, num_nodes, num_edges, feature_dim);

  {
    std::lock_guard<std::mutex> lock(st.mu);
    st.table[key] = p;
    st.dirty = true;
  }
  saveGraphPolicyCacheIfDirty();
  return p;
}

inline ops::MatMulIoPolicy makeGraphProjectionPolicy(const GraphSparseFriendlyPolicy& p, LoopStage stage) {
  if (!p.resident_aggregation || stage == LoopStage::kOneShot) {
    return ops::MatMulIoPolicy{};
  }
  if (stage == LoopStage::kStart) {
    return ops::makeMetalResidentStartPolicy();
  }
  if (stage == LoopStage::kRun) {
    return ops::makeMetalResidentRunPolicy();
  }
  return ops::makeMetalResidentFinishPolicy();
}

inline ops::VectorAddIoPolicy makeGraphUpdatePolicy(const GraphSparseFriendlyPolicy& p, LoopStage stage) {
  if (!p.resident_aggregation || stage == LoopStage::kOneShot) {
    return ops::VectorAddIoPolicy{};
  }
  if (stage == LoopStage::kStart) {
    return ops::makeMetalVectorResidentStartPolicy();
  }
  if (stage == LoopStage::kRun) {
    return ops::makeMetalVectorResidentRunPolicy();
  }
  return ops::makeMetalVectorResidentFinishPolicy();
}

}  // namespace lightning_core::models
