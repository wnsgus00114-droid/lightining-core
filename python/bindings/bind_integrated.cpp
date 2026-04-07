#include "bind_common.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

struct AttnSessionKey {
  std::size_t seq_len;
  std::size_t head_dim;
  bool causal;
  lc::Device device;

  bool operator==(const AttnSessionKey& other) const {
    return seq_len == other.seq_len && head_dim == other.head_dim && causal == other.causal && device == other.device;
  }
};

struct AttnSessionKeyHash {
  std::size_t operator()(const AttnSessionKey& k) const noexcept {
    const std::size_t h1 = std::hash<std::size_t>{}(k.seq_len);
    const std::size_t h2 = std::hash<std::size_t>{}(k.head_dim);
    const std::size_t h3 = std::hash<bool>{}(k.causal);
    const std::size_t h4 = std::hash<int>{}(static_cast<int>(k.device));
    return (((h1 * 1315423911u) ^ (h2 * 2654435761u)) ^ (h3 * 16777619u)) ^ h4;
  }
};

std::unordered_map<AttnSessionKey, std::shared_ptr<lc::AttentionSession>, AttnSessionKeyHash> g_attn_sessions;
std::mutex g_attn_sessions_mu;

struct ConvAttnGraphSessionKey {
  std::size_t batch;
  std::size_t in_channels;
  std::size_t in_h;
  std::size_t in_w;
  std::size_t out_channels;
  std::size_t seq_len;
  std::size_t head_dim;
  bool has_bias;
  lc::Device device;

  bool operator==(const ConvAttnGraphSessionKey& other) const {
    return batch == other.batch &&
           in_channels == other.in_channels &&
           in_h == other.in_h &&
           in_w == other.in_w &&
           out_channels == other.out_channels &&
           seq_len == other.seq_len &&
           head_dim == other.head_dim &&
           has_bias == other.has_bias &&
           device == other.device;
  }
};

struct ConvAttnGraphSessionKeyHash {
  std::size_t operator()(const ConvAttnGraphSessionKey& k) const noexcept {
    std::size_t h = std::hash<std::size_t>{}(k.batch);
    h = (h * 1315423911u) ^ std::hash<std::size_t>{}(k.in_channels);
    h = (h * 1315423911u) ^ std::hash<std::size_t>{}(k.in_h);
    h = (h * 1315423911u) ^ std::hash<std::size_t>{}(k.in_w);
    h = (h * 1315423911u) ^ std::hash<std::size_t>{}(k.out_channels);
    h = (h * 1315423911u) ^ std::hash<std::size_t>{}(k.seq_len);
    h = (h * 1315423911u) ^ std::hash<std::size_t>{}(k.head_dim);
    h = (h * 1315423911u) ^ std::hash<bool>{}(k.has_bias);
    h = (h * 1315423911u) ^ std::hash<int>{}(static_cast<int>(k.device));
    return h;
  }
};

struct ConvAttnGraphSession {
  lc::graph::GraphIR conv_graph;
  lc::graph::GraphIR attn_graph;
  lc::graph::GraphPlannerOptions options;
  std::size_t x_id{0};
  std::size_t w_id{0};
  std::size_t b_id{0};
  std::size_t conv_out_id{0};
  std::size_t q_id{0};
  std::size_t k_id{0};
  std::size_t v_id{0};
  std::size_t out_id{0};
  bool has_bias{false};
};

std::unordered_map<ConvAttnGraphSessionKey,
                   std::shared_ptr<ConvAttnGraphSession>,
                   ConvAttnGraphSessionKeyHash> g_conv_attn_graph_sessions;
std::mutex g_conv_attn_graph_sessions_mu;

enum class IntegratedExecutionMode {
  kEager = 0,
  kGraph
};

IntegratedExecutionMode parseIntegratedExecutionMode(const std::string& mode) {
  if (mode == "eager") {
    return IntegratedExecutionMode::kEager;
  }
  if (mode == "graph") {
    return IntegratedExecutionMode::kGraph;
  }
  throw std::invalid_argument("execution_mode must be 'eager' or 'graph'");
}

bool envFlagEnabled(const char* name) {
  if (name == nullptr) {
    return false;
  }
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }
  const char c0 = static_cast<char>(std::tolower(static_cast<unsigned char>(raw[0])));
  return c0 == '1' || c0 == 't' || c0 == 'y';
}

bool shouldPreferTinyCpuConvAttnChain(
    lc::Device requested_device,
    std::size_t batch,
    std::size_t in_channels,
    std::size_t out_channels,
    std::size_t kernel_h,
    std::size_t kernel_w,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t pad_h,
    std::size_t pad_w,
    std::size_t out_h,
    std::size_t out_w,
    std::size_t seq_len,
    std::size_t head_dim) {
  if (envFlagEnabled("LC_CONV_ATTN_TINY_CHAIN_DISABLE")) {
    return false;
  }
  if (requested_device != lc::Device::kMetal) {
    return false;
  }
  if (kernel_h != 3 || kernel_w != 3 || stride_h != 1 || stride_w != 1 || pad_h != 1 || pad_w != 1) {
    return false;
  }
  if (batch > 64 || in_channels > 16 || out_channels > 32) {
    return false;
  }
  const std::size_t conv_macs = batch * out_h * out_w * out_channels * in_channels * kernel_h * kernel_w;
  if (conv_macs > lc::ops::conv2dOneShotCpuCrossoverMacs()) {
    return false;
  }
  const std::size_t attn_work = seq_len * head_dim;
  if (attn_work == 0 || attn_work > 12288) {
    return false;
  }
  return true;
}

lc::graph::GraphPlannerOptions defaultGraphPlannerOptions(lc::Device preferred_device) {
  lc::graph::GraphPlannerOptions options;
  options.preferred_device = preferred_device;
  options.sync_policy.mode = lc::runtime::SyncMode::kAuto;
  options.sync_policy.trace_sync_boundary = false;
  options.separate_fallback_segments = true;
  options.insert_sync_on_device_change = true;
  return options;
}

const char* validationHintForPass(lc::graph::ValidationPass pass) {
  switch (pass) {
    case lc::graph::ValidationPass::kTensorSpec:
      return "check tensor shape/dtype/layout for the graph tensors";
    case lc::graph::ValidationPass::kSchemaArity:
      return "check operator input/output counts for the node schema";
    case lc::graph::ValidationPass::kTensorReference:
      return "check node tensor ids and feed dictionary coverage";
    case lc::graph::ValidationPass::kControlDependency:
      return "check control deps reference only prior node ids";
    case lc::graph::ValidationPass::kBackendCapability:
      return "operator is unavailable on current backend; try device='cpu' or execution_mode='eager'";
    default:
      return "check graph definition and backend capability";
  }
}

std::string formatFirstValidationIssue(const lc::graph::ValidationReport& report) {
  if (report.issues.empty()) {
    return "no validation issue details";
  }
  const lc::graph::ValidationIssue& issue = report.issues.front();
  std::ostringstream oss;
  oss << "pass=" << lc::graph::validationPassName(issue.pass)
      << ", status=" << lc::runtime::getErrorString(issue.status)
      << ", node_id=" << issue.node_id
      << ", tensor_id=" << issue.tensor_id
      << ", message=" << issue.message
      << ", hint=" << validationHintForPass(issue.pass);
  return oss.str();
}

void ensureGraphValidOrThrow(const lc::graph::GraphIR& graph, const char* graph_tag) {
  lc::graph::ValidationReport report;
  const lc::runtime::Status st = graph.validateWithReport(&report);
  if (st == lc::runtime::Status::kSuccess) {
    return;
  }
  std::ostringstream oss;
  oss << "graph validation failed (" << graph_tag << "): " << lc::runtime::getErrorString(st)
      << " | first_issue: " << formatFirstValidationIssue(report);
  throw std::runtime_error(oss.str());
}

void throwGraphExecuteFailure(const char* graph_tag,
                             lc::runtime::Status st,
                             const std::string& device_name) {
  std::ostringstream oss;
  oss << "graph execute_f32 failed (" << graph_tag << "): " << lc::runtime::getErrorString(st);
  if (st == lc::runtime::Status::kNotSupported) {
    oss << " | hint: selected backend is unsupported for this path on device='"
        << device_name << "'; try execution_mode='eager' or device='cpu'";
  } else if (st == lc::runtime::Status::kInvalidValue) {
    oss << " | hint: check feed shapes/numel and conv->attn dimensions (seq_len * head_dim)";
  } else {
    oss << " | hint: retry with execution_mode='eager' and compare using *_ab_report API";
  }
  throw std::runtime_error(oss.str());
}

void copyArrayToVector(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr,
                       std::vector<float>* out) {
  if (out == nullptr) {
    throw std::invalid_argument("internal error: null output vector");
  }
  const std::size_t n = static_cast<std::size_t>(arr.size());
  out->resize(n);
  if (n != 0) {
    std::memcpy(out->data(), arr.data(), n * sizeof(float));
  }
}

void copySpanToVector(const float* src, std::size_t n, std::vector<float>* out) {
  if (out == nullptr) {
    throw std::invalid_argument("internal error: null output vector");
  }
  out->resize(n);
  if (n != 0) {
    std::memcpy(out->data(), src, n * sizeof(float));
  }
}

std::shared_ptr<ConvAttnGraphSession> createConvAttnGraphSession(const ConvAttnGraphSessionKey& key) {
  auto session = std::make_shared<ConvAttnGraphSession>();
  session->options = defaultGraphPlannerOptions(key.device);
  session->has_bias = key.has_bias;

  lc::graph::TensorSpec x_spec;
  x_spec.shape = {
      static_cast<std::int64_t>(key.batch),
      static_cast<std::int64_t>(key.in_channels),
      static_cast<std::int64_t>(key.in_h),
      static_cast<std::int64_t>(key.in_w)};
  x_spec.dtype = lc::graph::DType::kFloat32;
  x_spec.layout = lc::Layout::kContiguous;

  lc::graph::TensorSpec w_spec;
  w_spec.shape = {
      static_cast<std::int64_t>(key.out_channels),
      static_cast<std::int64_t>(key.in_channels),
      3,
      3};
  w_spec.dtype = lc::graph::DType::kFloat32;
  w_spec.layout = lc::Layout::kContiguous;

  lc::graph::TensorSpec conv_out_spec;
  conv_out_spec.shape = {
      static_cast<std::int64_t>(key.batch),
      static_cast<std::int64_t>(key.out_channels),
      static_cast<std::int64_t>(key.in_h),
      static_cast<std::int64_t>(key.in_w)};
  conv_out_spec.dtype = lc::graph::DType::kFloat32;
  conv_out_spec.layout = lc::Layout::kContiguous;

  throwIfNotSuccess(session->conv_graph.addTensorSpec(x_spec, &session->x_id, "x", true));
  throwIfNotSuccess(session->conv_graph.addTensorSpec(w_spec, &session->w_id, "w", true));
  throwIfNotSuccess(session->conv_graph.addTensorSpec(conv_out_spec, &session->conv_out_id, "conv_out", false));

  std::vector<std::size_t> conv_inputs = {session->x_id, session->w_id};
  if (key.has_bias) {
    lc::graph::TensorSpec b_spec;
    b_spec.shape = {static_cast<std::int64_t>(key.out_channels)};
    b_spec.dtype = lc::graph::DType::kFloat32;
    b_spec.layout = lc::Layout::kContiguous;
    throwIfNotSuccess(session->conv_graph.addTensorSpec(b_spec, &session->b_id, "b", true));
    conv_inputs.push_back(session->b_id);
  }
  throwIfNotSuccess(session->conv_graph.addNode(
      lc::graph::OpKind::kConv2dNchw3x3s1p1, conv_inputs, {session->conv_out_id}));
  ensureGraphValidOrThrow(session->conv_graph, "conv2d_nchw3x3s1p1");

  lc::graph::TensorSpec attn_spec;
  attn_spec.shape = {
      static_cast<std::int64_t>(key.seq_len),
      static_cast<std::int64_t>(key.head_dim)};
  attn_spec.dtype = lc::graph::DType::kFloat32;
  attn_spec.layout = lc::Layout::kContiguous;

  throwIfNotSuccess(session->attn_graph.addTensorSpec(attn_spec, &session->q_id, "q", true));
  throwIfNotSuccess(session->attn_graph.addTensorSpec(attn_spec, &session->k_id, "k", true));
  throwIfNotSuccess(session->attn_graph.addTensorSpec(attn_spec, &session->v_id, "v", true));
  throwIfNotSuccess(session->attn_graph.addTensorSpec(attn_spec, &session->out_id, "out", false));
  throwIfNotSuccess(session->attn_graph.addNode(
      lc::graph::OpKind::kAttentionForward, {session->q_id, session->k_id, session->v_id}, {session->out_id}));
  ensureGraphValidOrThrow(session->attn_graph, "attention_forward");

  return session;
}

std::shared_ptr<ConvAttnGraphSession> getOrCreateConvAttnGraphSession(const ConvAttnGraphSessionKey& key) {
  thread_local bool tls_valid = false;
  thread_local ConvAttnGraphSessionKey tls_key{0, 0, 0, 0, 0, 0, 0, false, lc::Device::kCPU};
  thread_local std::shared_ptr<ConvAttnGraphSession> tls_session;

  if (tls_valid && tls_session && tls_key == key) {
    return tls_session;
  }

  {
    std::lock_guard<std::mutex> lock(g_conv_attn_graph_sessions_mu);
    auto it = g_conv_attn_graph_sessions.find(key);
    if (it != g_conv_attn_graph_sessions.end()) {
      tls_key = key;
      tls_session = it->second;
      tls_valid = true;
      return it->second;
    }
  }

  auto session = createConvAttnGraphSession(key);
  {
    std::lock_guard<std::mutex> lock(g_conv_attn_graph_sessions_mu);
    g_conv_attn_graph_sessions[key] = session;
  }
  tls_key = key;
  tls_session = session;
  tls_valid = true;
  return session;
}

struct DiffStats {
  double max_abs_diff{0.0};
  double mean_abs_diff{0.0};
  double max_rel_diff{0.0};
  bool allclose{true};
};

DiffStats computeDiffStats(const float* ref,
                           const float* cand,
                           std::size_t n,
                           double atol,
                           double rtol) {
  DiffStats out;
  if (n == 0) {
    return out;
  }
  double sum_abs = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double r = static_cast<double>(ref[i]);
    const double c = static_cast<double>(cand[i]);
    const double abs_diff = std::fabs(r - c);
    const double rel_diff = abs_diff / (std::fabs(r) + 1e-12);
    out.max_abs_diff = std::max(out.max_abs_diff, abs_diff);
    out.max_rel_diff = std::max(out.max_rel_diff, rel_diff);
    sum_abs += abs_diff;
    if (abs_diff > (atol + (rtol * std::fabs(r)))) {
      out.allclose = false;
    }
  }
  out.mean_abs_diff = sum_abs / static_cast<double>(n);
  return out;
}

std::shared_ptr<lc::AttentionSession> getOrCreateAttentionSession(std::size_t seq_len,
                                                                  std::size_t head_dim,
                                                                  bool causal,
                                                                  lc::Device device) {
  AttnSessionKey key{seq_len, head_dim, causal, device};
  thread_local bool tls_valid = false;
  thread_local AttnSessionKey tls_key{0, 0, false, lc::Device::kCPU};
  thread_local std::shared_ptr<lc::AttentionSession> tls_session;

  if (tls_valid && tls_session && tls_key == key) {
    return tls_session;
  }
  {
    std::lock_guard<std::mutex> lock(g_attn_sessions_mu);
    auto it = g_attn_sessions.find(key);
    if (it != g_attn_sessions.end()) {
      tls_key = key;
      tls_session = it->second;
      tls_valid = true;
      return it->second;
    }
  }

  auto session = std::make_shared<lc::AttentionSession>(lc::AttentionConfig{seq_len, head_dim, causal}, device);
  {
    std::lock_guard<std::mutex> lock(g_attn_sessions_mu);
    g_attn_sessions[key] = session;
  }
  tls_key = key;
  tls_session = session;
  tls_valid = true;
  return session;
}

py::array_t<float> conv2dNchwDirect(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
    py::object bias_obj,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t pad_h,
    std::size_t pad_w,
  const std::string& device_name,
  bool apply_relu = false) {
  auto xbuf = x.request();
  auto wbuf = w.request();
  if (xbuf.ndim != 4 || wbuf.ndim != 4) {
    throw std::invalid_argument("x and w must be 4D arrays (NCHW and OIHW)");
  }

  const std::size_t batch = static_cast<std::size_t>(xbuf.shape[0]);
  const std::size_t in_channels = static_cast<std::size_t>(xbuf.shape[1]);
  const std::size_t in_h = static_cast<std::size_t>(xbuf.shape[2]);
  const std::size_t in_w = static_cast<std::size_t>(xbuf.shape[3]);

  const std::size_t out_channels = static_cast<std::size_t>(wbuf.shape[0]);
  const std::size_t w_in_channels = static_cast<std::size_t>(wbuf.shape[1]);
  const std::size_t kernel_h = static_cast<std::size_t>(wbuf.shape[2]);
  const std::size_t kernel_w = static_cast<std::size_t>(wbuf.shape[3]);

  if (in_channels != w_in_channels) {
    throw std::invalid_argument("x channels and w in_channels mismatch");
  }
  if (stride_h == 0 || stride_w == 0) {
    throw std::invalid_argument("stride must be > 0");
  }
  if (in_h + (2 * pad_h) < kernel_h || in_w + (2 * pad_w) < kernel_w) {
    throw std::invalid_argument("invalid padding/kernel for input size");
  }

  const std::size_t out_h = (in_h + (2 * pad_h) - kernel_h) / stride_h + 1;
  const std::size_t out_w = (in_w + (2 * pad_w) - kernel_w) / stride_w + 1;
  py::array_t<float> out({batch, out_channels, out_h, out_w});

  const float* bias_ptr = nullptr;
  py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
  if (!bias_obj.is_none()) {
    bias_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(bias_obj);
    if (static_cast<std::size_t>(bias_arr.size()) != out_channels) {
      throw std::invalid_argument("bias shape mismatch");
    }
    bias_ptr = bias_arr.data();
  }

  throwIfNotSuccess(lc::ops::conv2dNchw<float>(
      x.data(),
      w.data(),
      bias_ptr,
      out.mutable_data(),
      batch,
      in_channels,
      in_h,
      in_w,
      out_channels,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      parseDevice(device_name),
      apply_relu));
  return out;
}

void conv2dNchwIntoDirect(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
    py::object bias_obj,
    py::array_t<float, py::array::c_style | py::array::forcecast>& out,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t pad_h,
    std::size_t pad_w,
    const std::string& device_name,
    bool apply_relu = false) {
  auto xbuf = x.request();
  auto wbuf = w.request();
  auto obuf = out.request();
  if (xbuf.ndim != 4 || wbuf.ndim != 4 || obuf.ndim != 4) {
    throw std::invalid_argument("x, w, out must be 4D arrays");
  }

  const std::size_t batch = static_cast<std::size_t>(xbuf.shape[0]);
  const std::size_t in_channels = static_cast<std::size_t>(xbuf.shape[1]);
  const std::size_t in_h = static_cast<std::size_t>(xbuf.shape[2]);
  const std::size_t in_w = static_cast<std::size_t>(xbuf.shape[3]);

  const std::size_t out_channels = static_cast<std::size_t>(wbuf.shape[0]);
  const std::size_t w_in_channels = static_cast<std::size_t>(wbuf.shape[1]);
  const std::size_t kernel_h = static_cast<std::size_t>(wbuf.shape[2]);
  const std::size_t kernel_w = static_cast<std::size_t>(wbuf.shape[3]);

  if (in_channels != w_in_channels) {
    throw std::invalid_argument("x channels and w in_channels mismatch");
  }
  if (stride_h == 0 || stride_w == 0) {
    throw std::invalid_argument("stride must be > 0");
  }
  if (in_h + (2 * pad_h) < kernel_h || in_w + (2 * pad_w) < kernel_w) {
    throw std::invalid_argument("invalid padding/kernel for input size");
  }

  const std::size_t out_h = (in_h + (2 * pad_h) - kernel_h) / stride_h + 1;
  const std::size_t out_w = (in_w + (2 * pad_w) - kernel_w) / stride_w + 1;
  if (static_cast<std::size_t>(obuf.shape[0]) != batch ||
      static_cast<std::size_t>(obuf.shape[1]) != out_channels ||
      static_cast<std::size_t>(obuf.shape[2]) != out_h ||
      static_cast<std::size_t>(obuf.shape[3]) != out_w) {
    throw std::invalid_argument("out shape mismatch");
  }

  const float* bias_ptr = nullptr;
  py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
  if (!bias_obj.is_none()) {
    bias_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(bias_obj);
    if (static_cast<std::size_t>(bias_arr.size()) != out_channels) {
      throw std::invalid_argument("bias shape mismatch");
    }
    bias_ptr = bias_arr.data();
  }

	  throwIfNotSuccess(lc::ops::conv2dNchw<float>(
	      x.data(),
	      w.data(),
	      bias_ptr,
	      out.mutable_data(),
      batch,
      in_channels,
      in_h,
      in_w,
      out_channels,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
	      parseDevice(device_name),
	      apply_relu));
}

void attentionIntoDirect(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& q,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& k,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& v,
    py::array_t<float, py::array::c_style | py::array::forcecast>& out,
    std::size_t seq_len,
    std::size_t head_dim,
    bool causal,
    const std::string& device_name) {
  const std::size_t expected = seq_len * head_dim;
  requireSize(q, expected, "q");
  requireSize(k, expected, "k");
  requireSize(v, expected, "v");
  requireSize(out, expected, "out");
  const lc::Device device = parseDevice(device_name);
  auto session = getOrCreateAttentionSession(seq_len, head_dim, causal, device);
  throwIfNotSuccess(session->forward(q.data(), k.data(), v.data(), out.mutable_data()));
}

void convAttentionTorchstrongNchwIntoDirect(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
    py::object bias_obj,
  py::array_t<float, py::array::c_style | py::array::forcecast>& out,
    std::size_t seq_len,
    std::size_t head_dim,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t pad_h,
    std::size_t pad_w,
    const std::string& device_name) {
  const std::size_t need = seq_len * head_dim;
  requireSize(out, need, "out");

  auto xbuf = x.request();
  auto wbuf = w.request();
  if (xbuf.ndim != 4 || wbuf.ndim != 4) {
    throw std::invalid_argument("x and w must be 4D arrays (NCHW and OIHW)");
  }

  const std::size_t batch = static_cast<std::size_t>(xbuf.shape[0]);
  const std::size_t in_channels = static_cast<std::size_t>(xbuf.shape[1]);
  const std::size_t in_h = static_cast<std::size_t>(xbuf.shape[2]);
  const std::size_t in_w = static_cast<std::size_t>(xbuf.shape[3]);

  const std::size_t out_channels = static_cast<std::size_t>(wbuf.shape[0]);
  const std::size_t w_in_channels = static_cast<std::size_t>(wbuf.shape[1]);
  const std::size_t kernel_h = static_cast<std::size_t>(wbuf.shape[2]);
  const std::size_t kernel_w = static_cast<std::size_t>(wbuf.shape[3]);

  if (in_channels != w_in_channels) {
    throw std::invalid_argument("x channels and w in_channels mismatch");
  }
  if (stride_h == 0 || stride_w == 0) {
    throw std::invalid_argument("stride must be > 0");
  }
  if (in_h + (2 * pad_h) < kernel_h || in_w + (2 * pad_w) < kernel_w) {
    throw std::invalid_argument("invalid padding/kernel for input size");
  }

  const std::size_t out_h = (in_h + (2 * pad_h) - kernel_h) / stride_h + 1;
  const std::size_t out_w = (in_w + (2 * pad_w) - kernel_w) / stride_w + 1;
  const std::size_t conv_size = batch * out_channels * out_h * out_w;
  const lc::Device requested_device = parseDevice(device_name);
  const bool tiny_cpu_chain = shouldPreferTinyCpuConvAttnChain(
      requested_device,
      batch,
      in_channels,
      out_channels,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      out_h,
      out_w,
      seq_len,
      head_dim);
  const lc::Device attn_device = tiny_cpu_chain ? lc::Device::kCPU : requested_device;

  const float* bias_ptr = nullptr;
  py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
  if (!bias_obj.is_none()) {
    bias_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(bias_obj);
    if (static_cast<std::size_t>(bias_arr.size()) != out_channels) {
      throw std::invalid_argument("bias shape mismatch");
    }
    bias_ptr = bias_arr.data();
  }

  thread_local std::vector<float> conv_tmp;
  if (conv_tmp.size() < conv_size) {
    conv_tmp.resize(conv_size);
  }
  throwIfNotSuccess(lc::ops::conv2dNchw<float>(
      x.data(),
      w.data(),
      bias_ptr,
      conv_tmp.data(),
      batch,
      in_channels,
      in_h,
      in_w,
      out_channels,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      requested_device,
      true));

  const std::size_t total = need * 3;
  const float* conv_ptr = conv_tmp.data();
  auto session = getOrCreateAttentionSession(seq_len, head_dim, false, attn_device);

  if (conv_size >= total) {
    throwIfNotSuccess(session->forward(conv_ptr, conv_ptr + need, conv_ptr + (2 * need), out.mutable_data()));
    return;
  }

  thread_local std::vector<float> packed;
  if (packed.size() < total) {
    packed.resize(total);
  }
  std::size_t copied = std::min(conv_size, total);
  std::memcpy(packed.data(), conv_ptr, copied * sizeof(float));
  while (copied < total) {
    const std::size_t chunk = std::min(copied, total - copied);
    std::memcpy(packed.data() + copied, packed.data(), chunk * sizeof(float));
    copied += chunk;
  }
  throwIfNotSuccess(session->forward(packed.data(), packed.data() + need, packed.data() + (2 * need), out.mutable_data()));
}

void convAttentionTorchstrongNchwIntoWithMode(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
    py::object bias_obj,
    py::array_t<float, py::array::c_style | py::array::forcecast>& out,
    std::size_t seq_len,
    std::size_t head_dim,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t pad_h,
    std::size_t pad_w,
    const std::string& device_name,
    const std::string& execution_mode);

void convAttentionTorchstrongNchwIntoGraph(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
    py::object bias_obj,
    py::array_t<float, py::array::c_style | py::array::forcecast>& out,
    std::size_t seq_len,
    std::size_t head_dim,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t pad_h,
    std::size_t pad_w,
    const std::string& device_name) {
  const std::size_t need = seq_len * head_dim;
  requireSize(out, need, "out");

  auto xbuf = x.request();
  auto wbuf = w.request();
  if (xbuf.ndim != 4 || wbuf.ndim != 4) {
    throw std::invalid_argument("x and w must be 4D arrays (NCHW and OIHW)");
  }

  const std::size_t batch = static_cast<std::size_t>(xbuf.shape[0]);
  const std::size_t in_channels = static_cast<std::size_t>(xbuf.shape[1]);
  const std::size_t in_h = static_cast<std::size_t>(xbuf.shape[2]);
  const std::size_t in_w = static_cast<std::size_t>(xbuf.shape[3]);

  const std::size_t out_channels = static_cast<std::size_t>(wbuf.shape[0]);
  const std::size_t w_in_channels = static_cast<std::size_t>(wbuf.shape[1]);
  const std::size_t kernel_h = static_cast<std::size_t>(wbuf.shape[2]);
  const std::size_t kernel_w = static_cast<std::size_t>(wbuf.shape[3]);
  if (in_channels != w_in_channels) {
    throw std::invalid_argument("x channels and w in_channels mismatch");
  }
  if (stride_h != 1 || stride_w != 1 || pad_h != 1 || pad_w != 1 || kernel_h != 3 || kernel_w != 3) {
    throw std::invalid_argument("graph mode currently supports conv2d_nchw3x3s1p1 only");
  }

  const std::size_t out_h = (in_h + (2 * pad_h) - kernel_h) / stride_h + 1;
  const std::size_t out_w = (in_w + (2 * pad_w) - kernel_w) / stride_w + 1;
  const std::size_t conv_size = batch * out_channels * out_h * out_w;
  if (conv_size == 0) {
    throw std::invalid_argument("invalid conv shape");
  }

  const lc::Device device = parseDevice(device_name);
  py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
  const bool has_bias = !bias_obj.is_none();
  if (has_bias) {
    bias_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(bias_obj);
    if (static_cast<std::size_t>(bias_arr.size()) != out_channels) {
      throw std::invalid_argument("bias shape mismatch");
    }
  }

  ConvAttnGraphSessionKey graph_key{
      batch,
      in_channels,
      in_h,
      in_w,
      out_channels,
      seq_len,
      head_dim,
      has_bias,
      device};
  auto graph_session = getOrCreateConvAttnGraphSession(graph_key);

  thread_local std::unordered_map<std::size_t, std::vector<float>> conv_feeds;
  thread_local std::unordered_map<std::size_t, std::vector<float>> conv_values;
  conv_feeds.clear();
  conv_values.clear();

  copyArrayToVector(x, &conv_feeds[graph_session->x_id]);
  copyArrayToVector(w, &conv_feeds[graph_session->w_id]);
  if (has_bias) {
    copyArrayToVector(bias_arr, &conv_feeds[graph_session->b_id]);
  }
  const lc::runtime::Status conv_exec_status =
      graph_session->conv_graph.executeF32(graph_session->options, conv_feeds, &conv_values, nullptr, nullptr);
  if (conv_exec_status != lc::runtime::Status::kSuccess) {
    throwGraphExecuteFailure("conv2d_nchw3x3s1p1", conv_exec_status, device_name);
  }
  auto conv_it = conv_values.find(graph_session->conv_out_id);
  if (conv_it == conv_values.end()) {
    throw std::runtime_error("graph conv output missing");
  }
  const std::vector<float>& conv_flat = conv_it->second;

  const std::size_t total = need * 3;
  thread_local std::vector<float> packed;
  if (packed.size() < total) {
    packed.resize(total);
  }
  std::size_t copied = std::min(conv_flat.size(), total);
  if (copied == 0 && total != 0) {
    throw std::runtime_error("graph conv output is empty");
  }
  std::memcpy(packed.data(), conv_flat.data(), copied * sizeof(float));
  while (copied < total) {
    const std::size_t chunk = std::min(copied, total - copied);
    std::memcpy(packed.data() + copied, packed.data(), chunk * sizeof(float));
    copied += chunk;
  }

  thread_local std::unordered_map<std::size_t, std::vector<float>> attn_feeds;
  thread_local std::unordered_map<std::size_t, std::vector<float>> attn_values;
  attn_feeds.clear();
  attn_values.clear();

  copySpanToVector(packed.data(), need, &attn_feeds[graph_session->q_id]);
  copySpanToVector(packed.data() + need, need, &attn_feeds[graph_session->k_id]);
  copySpanToVector(packed.data() + (2 * need), need, &attn_feeds[graph_session->v_id]);

  const lc::runtime::Status attn_exec_status =
      graph_session->attn_graph.executeF32(graph_session->options, attn_feeds, &attn_values, nullptr, nullptr);
  if (attn_exec_status != lc::runtime::Status::kSuccess) {
    throwGraphExecuteFailure("attention_forward", attn_exec_status, device_name);
  }
  auto out_it = attn_values.find(graph_session->out_id);
  if (out_it == attn_values.end() || out_it->second.size() != need) {
    throw std::runtime_error("graph attention output missing or size mismatch");
  }
  std::memcpy(out.mutable_data(), out_it->second.data(), need * sizeof(float));
}

py::dict convAttentionTorchstrongNchwAbReport(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
    py::object bias_obj,
    std::size_t seq_len,
    std::size_t head_dim,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t pad_h,
    std::size_t pad_w,
    const std::string& device_name,
    std::size_t warmup,
    std::size_t repeat,
    double atol,
    double rtol) {
  if (repeat == 0) {
    throw std::invalid_argument("repeat must be > 0");
  }
  if (atol < 0.0 || rtol < 0.0) {
    throw std::invalid_argument("atol/rtol must be >= 0");
  }

  const std::size_t need = seq_len * head_dim;
  auto eager_out = py::array_t<float, py::array::c_style | py::array::forcecast>(need);
  auto graph_out = py::array_t<float, py::array::c_style | py::array::forcecast>(need);

  auto run_mode = [&](const std::string& mode,
                      py::array_t<float, py::array::c_style | py::array::forcecast>& out) -> double {
    for (std::size_t i = 0; i < warmup; ++i) {
      convAttentionTorchstrongNchwIntoWithMode(
          x, w, bias_obj, out, seq_len, head_dim, stride_h, stride_w, pad_h, pad_w, device_name, mode);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < repeat; ++i) {
      convAttentionTorchstrongNchwIntoWithMode(
          x, w, bias_obj, out, seq_len, head_dim, stride_h, stride_w, pad_h, pad_w, device_name, mode);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return elapsed_ms / static_cast<double>(repeat);
  };

  const double eager_ms = run_mode("eager", eager_out);
  const double graph_ms = run_mode("graph", graph_out);
  const DiffStats diff = computeDiffStats(eager_out.data(), graph_out.data(), need, atol, rtol);

  py::dict report;
  report["seq_len"] = seq_len;
  report["head_dim"] = head_dim;
  report["warmup"] = warmup;
  report["repeat"] = repeat;
  report["device"] = device_name;
  report["allclose"] = diff.allclose;
  report["max_abs_diff"] = diff.max_abs_diff;
  report["mean_abs_diff"] = diff.mean_abs_diff;
  report["max_rel_diff"] = diff.max_rel_diff;
  report["atol"] = atol;
  report["rtol"] = rtol;
  report["eager_ms"] = eager_ms;
  report["graph_ms"] = graph_ms;
  report["graph_over_eager"] = eager_ms > 0.0 ? (graph_ms / eager_ms) : 0.0;
  report["eager_over_graph"] = graph_ms > 0.0 ? (eager_ms / graph_ms) : 0.0;
  if (graph_ms < eager_ms) {
    report["winner"] = "graph";
  } else if (eager_ms < graph_ms) {
    report["winner"] = "eager";
  } else {
    report["winner"] = "tie";
  }
  return report;
}

void convAttentionTorchstrongNchwIntoWithMode(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
    py::object bias_obj,
    py::array_t<float, py::array::c_style | py::array::forcecast>& out,
    std::size_t seq_len,
    std::size_t head_dim,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t pad_h,
    std::size_t pad_w,
    const std::string& device_name,
    const std::string& execution_mode) {
  switch (parseIntegratedExecutionMode(execution_mode)) {
    case IntegratedExecutionMode::kEager:
      convAttentionTorchstrongNchwIntoDirect(
          x, w, bias_obj, out, seq_len, head_dim, stride_h, stride_w, pad_h, pad_w, device_name);
      return;
    case IntegratedExecutionMode::kGraph:
      convAttentionTorchstrongNchwIntoGraph(
          x, w, bias_obj, out, seq_len, head_dim, stride_h, stride_w, pad_h, pad_w, device_name);
      return;
    default:
      throw std::invalid_argument("unsupported execution mode");
  }
}

}  // namespace

void bindIntegrated(py::module_& m) {
  auto bind_api = [](py::module_& mod, bool legacy_prefix) {
    const char* clear_name = legacy_prefix ? "clear_integrated_attention_session_cache" : "clear_attention_session_cache";
    const char* conv_relu_name = legacy_prefix ? "lightning_conv_relu_nchw" : "conv_relu_nchw";
    const char* conv_relu_into_name = legacy_prefix ? "lightning_conv_relu_nchw_into" : "conv_relu_nchw_into";
    const char* attention_name = legacy_prefix ? "lightning_attention" : "attention";
    const char* attention_into_name = legacy_prefix ? "lightning_attention_into" : "attention_into";
    const char* conv_attn_name = legacy_prefix ? "lightning_conv_attention_torchstrong_nchw" : "conv_attention_torchstrong_nchw";
    const char* conv_attn_into_name = legacy_prefix ? "lightning_conv_attention_torchstrong_nchw_into" : "conv_attention_torchstrong_nchw_into";
    const char* conv_attn_ab_report_name = legacy_prefix
        ? "lightning_conv_attention_torchstrong_nchw_ab_report"
        : "conv_attention_torchstrong_nchw_ab_report";

    mod.def(clear_name, []() {
      {
        std::lock_guard<std::mutex> lock(g_attn_sessions_mu);
        g_attn_sessions.clear();
      }
      {
        std::lock_guard<std::mutex> lock(g_conv_attn_graph_sessions_mu);
        g_conv_attn_graph_sessions.clear();
      }
    });

    mod.def(conv_relu_name,
            [](const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
               py::object bias_obj,
               std::size_t stride_h,
               std::size_t stride_w,
               std::size_t pad_h,
               std::size_t pad_w,
               const std::string& device_name) {
              auto out = conv2dNchwDirect(x, w, bias_obj, stride_h, stride_w, pad_h, pad_w, device_name, true);
              return out;
            },
            py::arg("x"),
            py::arg("w"),
            py::arg("bias") = py::none(),
            py::arg("stride_h") = 1,
            py::arg("stride_w") = 1,
            py::arg("pad_h") = 0,
            py::arg("pad_w") = 0,
            py::arg("device") = "metal");

    mod.def(conv_relu_into_name,
            [](const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
               py::object bias_obj,
               py::array_t<float, py::array::c_style | py::array::forcecast>& out,
               std::size_t stride_h,
               std::size_t stride_w,
               std::size_t pad_h,
               std::size_t pad_w,
               const std::string& device_name) {
              conv2dNchwIntoDirect(x, w, bias_obj, out, stride_h, stride_w, pad_h, pad_w, device_name, true);
            },
            py::arg("x"),
            py::arg("w"),
            py::arg("bias"),
            py::arg("out"),
            py::arg("stride_h") = 1,
            py::arg("stride_w") = 1,
            py::arg("pad_h") = 0,
            py::arg("pad_w") = 0,
            py::arg("device") = "metal");

    mod.def(attention_name,
            [](const py::array_t<float, py::array::c_style | py::array::forcecast>& q,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& k,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& v,
               std::size_t seq_len,
               std::size_t head_dim,
               bool causal,
               const std::string& device_name) {
              const std::size_t expected = seq_len * head_dim;
              auto out = py::array_t<float, py::array::c_style | py::array::forcecast>(expected);
              attentionIntoDirect(q, k, v, out, seq_len, head_dim, causal, device_name);
              return out;
            },
            py::arg("q"),
            py::arg("k"),
            py::arg("v"),
            py::arg("seq_len"),
            py::arg("head_dim"),
            py::arg("causal") = false,
            py::arg("device") = "metal");

    mod.def(attention_into_name,
            [](const py::array_t<float, py::array::c_style | py::array::forcecast>& q,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& k,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& v,
               py::array_t<float, py::array::c_style | py::array::forcecast>& out,
               std::size_t seq_len,
               std::size_t head_dim,
               bool causal,
               const std::string& device_name) {
              attentionIntoDirect(q, k, v, out, seq_len, head_dim, causal, device_name);
            },
            py::arg("q"),
            py::arg("k"),
            py::arg("v"),
            py::arg("out"),
            py::arg("seq_len"),
            py::arg("head_dim"),
            py::arg("causal") = false,
            py::arg("device") = "metal");

    mod.def(conv_attn_name,
            [](const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
               py::object bias_obj,
               std::size_t seq_len,
               std::size_t head_dim,
               std::size_t stride_h,
               std::size_t stride_w,
               std::size_t pad_h,
               std::size_t pad_w,
               const std::string& device_name,
               const std::string& execution_mode) {
              const std::size_t need = seq_len * head_dim;
              auto out = py::array_t<float, py::array::c_style | py::array::forcecast>(need);
              convAttentionTorchstrongNchwIntoWithMode(
                  x,
                  w,
                  bias_obj,
                  out,
                  seq_len,
                  head_dim,
                  stride_h,
                  stride_w,
                  pad_h,
                  pad_w,
                  device_name,
                  execution_mode);
              return out;
            },
            py::arg("x"),
            py::arg("w"),
            py::arg("bias"),
            py::arg("seq_len"),
            py::arg("head_dim"),
            py::arg("stride_h") = 1,
            py::arg("stride_w") = 1,
            py::arg("pad_h") = 0,
            py::arg("pad_w") = 0,
            py::arg("device") = "metal",
            py::arg("execution_mode") = "eager");

        mod.def(conv_attn_into_name,
            [](const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
               py::object bias_obj,
               py::array_t<float, py::array::c_style | py::array::forcecast>& out,
               std::size_t seq_len,
               std::size_t head_dim,
               std::size_t stride_h,
               std::size_t stride_w,
               std::size_t pad_h,
               std::size_t pad_w,
               const std::string& device_name,
               const std::string& execution_mode) {
              convAttentionTorchstrongNchwIntoWithMode(
                x,
                w,
                bias_obj,
                out,
                seq_len,
                head_dim,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                device_name,
                execution_mode);
            },
            py::arg("x"),
            py::arg("w"),
            py::arg("bias"),
            py::arg("out"),
            py::arg("seq_len"),
            py::arg("head_dim"),
            py::arg("stride_h") = 1,
            py::arg("stride_w") = 1,
            py::arg("pad_h") = 0,
            py::arg("pad_w") = 0,
            py::arg("device") = "metal",
            py::arg("execution_mode") = "eager");

        mod.def(conv_attn_ab_report_name,
            [](const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
               py::object bias_obj,
               std::size_t seq_len,
               std::size_t head_dim,
               std::size_t stride_h,
               std::size_t stride_w,
               std::size_t pad_h,
               std::size_t pad_w,
               const std::string& device_name,
               std::size_t warmup,
               std::size_t repeat,
               double atol,
               double rtol) {
              return convAttentionTorchstrongNchwAbReport(
                  x,
                  w,
                  bias_obj,
                  seq_len,
                  head_dim,
                  stride_h,
                  stride_w,
                  pad_h,
                  pad_w,
                  device_name,
                  warmup,
                  repeat,
                  atol,
                  rtol);
            },
            py::arg("x"),
            py::arg("w"),
            py::arg("bias"),
            py::arg("seq_len"),
            py::arg("head_dim"),
            py::arg("stride_h") = 1,
            py::arg("stride_w") = 1,
            py::arg("pad_h") = 0,
            py::arg("pad_w") = 0,
            py::arg("device") = "metal",
            py::arg("warmup") = 1,
            py::arg("repeat") = 5,
            py::arg("atol") = 1e-4,
            py::arg("rtol") = 1e-4);
  };

  bind_api(m, true);

  py::module_ api = m.def_submodule("api", "High-level Python API on top of Lightning Core runtime");
  bind_api(api, false);
}
