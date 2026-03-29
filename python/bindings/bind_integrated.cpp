#include "bind_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
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
  const lc::Device device = parseDevice(device_name);
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
      device,
      true));

  const std::size_t total = need * 3;
  const float* conv_ptr = conv_tmp.data();
  auto session = getOrCreateAttentionSession(seq_len, head_dim, false, device);

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

    mod.def(clear_name, []() {
      std::lock_guard<std::mutex> lock(g_attn_sessions_mu);
      g_attn_sessions.clear();
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
               const std::string& device_name) {
              const std::size_t need = seq_len * head_dim;
              auto out = py::array_t<float, py::array::c_style | py::array::forcecast>(need);
              convAttentionTorchstrongNchwIntoDirect(
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
                  device_name);
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
            py::arg("device") = "metal");

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
               const std::string& device_name) {
              convAttentionTorchstrongNchwIntoDirect(
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
                device_name);
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
            py::arg("device") = "metal");
  };

  bind_api(m, true);

  py::module_ api = m.def_submodule("api", "High-level Python API on top of Lightning Core runtime");
  bind_api(api, false);
}
