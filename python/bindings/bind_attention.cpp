#include "bind_common.hpp"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace {

struct AttentionSessionKey {
  std::size_t seq_len;
  std::size_t head_dim;
  bool causal;
  lc::Device device;

  bool operator==(const AttentionSessionKey& other) const {
    return seq_len == other.seq_len &&
           head_dim == other.head_dim &&
           causal == other.causal &&
           device == other.device;
  }
};

struct AttentionSessionKeyHash {
  std::size_t operator()(const AttentionSessionKey& k) const noexcept {
    const std::size_t h1 = std::hash<std::size_t>{}(k.seq_len);
    const std::size_t h2 = std::hash<std::size_t>{}(k.head_dim);
    const std::size_t h3 = std::hash<bool>{}(k.causal);
    const std::size_t h4 = std::hash<int>{}(static_cast<int>(k.device));
    return (((h1 * 1315423911u) ^ (h2 * 2654435761u)) ^ (h3 * 16777619u)) ^ h4;
  }
};

std::unordered_map<AttentionSessionKey, std::shared_ptr<lc::AttentionSession>, AttentionSessionKeyHash> g_attention_sessions;
std::mutex g_attention_sessions_mu;

std::shared_ptr<lc::AttentionSession> getOrCreateAttentionSession(std::size_t seq_len,
                                                                  std::size_t head_dim,
                                                                  bool causal,
                                                                  lc::Device device) {
  AttentionSessionKey key{seq_len, head_dim, causal, device};
  thread_local bool tls_valid = false;
  thread_local AttentionSessionKey tls_key{0, 0, false, lc::Device::kCPU};
  thread_local std::shared_ptr<lc::AttentionSession> tls_session;

  if (tls_valid && tls_session && tls_key == key) {
    return tls_session;
  }

  {
    std::lock_guard<std::mutex> lock(g_attention_sessions_mu);
    auto it = g_attention_sessions.find(key);
    if (it != g_attention_sessions.end()) {
      tls_key = key;
      tls_session = it->second;
      tls_valid = true;
      return it->second;
    }
  }

  auto session = std::make_shared<lc::AttentionSession>(lc::AttentionConfig{seq_len, head_dim, causal}, device);
  {
    std::lock_guard<std::mutex> lock(g_attention_sessions_mu);
    g_attention_sessions[key] = session;
  }
  tls_key = key;
  tls_session = session;
  tls_valid = true;
  return session;
}

void attentionForwardIntoCachedSession(const float* q,
                                       const float* k,
                                       const float* v,
                                       float* out,
                                       std::size_t seq_len,
                                       std::size_t head_dim,
                                       bool causal,
                                       lc::Device device,
                                       const lc::AttentionIoPolicy* policy) {
  auto session = getOrCreateAttentionSession(seq_len, head_dim, causal, device);
  if (policy == nullptr) {
    throwIfNotSuccess(session->forward(q, k, v, out));
    return;
  }
  throwIfNotSuccess(session->forwardWithPolicy(q, k, v, out, *policy));
}

std::vector<float> attentionForwardVec(const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v,
                                       std::size_t seq_len,
                                       std::size_t head_dim,
                                       bool causal,
                                       lc::Device device,
                                       const lc::AttentionIoPolicy* policy) {
  const std::size_t expected = seq_len * head_dim;
  if (q.size() != expected || k.size() != expected || v.size() != expected) {
    throw std::invalid_argument("q/k/v length must match seq_len * head_dim");
  }

  std::vector<float> out(expected, 0.0f);
  attentionForwardIntoCachedSession(
      q.data(), k.data(), v.data(), out.data(), seq_len, head_dim, causal, device, policy);
  return out;
}

void requireAttention2D(const py::buffer_info& info,
                        const char* name,
                        std::size_t* seq_len_out,
                        std::size_t* head_dim_out) {
  if (seq_len_out == nullptr || head_dim_out == nullptr) {
    throw std::invalid_argument("internal error: null output shape pointer");
  }
  if (info.ndim != 2 || info.shape.size() != 2) {
    throw std::invalid_argument(std::string(name) + " must be a 2D float32 array shaped [seq_len, head_dim]");
  }
  *seq_len_out = static_cast<std::size_t>(info.shape[0]);
  *head_dim_out = static_cast<std::size_t>(info.shape[1]);
}

}  // namespace

void bindAttention(py::module_& m) {
  m.def("clear_attention2d_session_cache", []() {
    std::lock_guard<std::mutex> lock(g_attention_sessions_mu);
    g_attention_sessions.clear();
  });

  py::class_<lc::AttentionIoPolicy>(m, "AttentionIoPolicy")
      .def(py::init<>())
      .def_readwrite("upload_q", &lc::AttentionIoPolicy::upload_q)
      .def_readwrite("upload_k", &lc::AttentionIoPolicy::upload_k)
      .def_readwrite("upload_v", &lc::AttentionIoPolicy::upload_v)
      .def_readwrite("upload_target", &lc::AttentionIoPolicy::upload_target)
      .def_readwrite("download_out", &lc::AttentionIoPolicy::download_out)
      .def_readwrite("download_v", &lc::AttentionIoPolicy::download_v)
      .def_readwrite("synchronize", &lc::AttentionIoPolicy::synchronize)
      .def_readwrite("loss_every", &lc::AttentionIoPolicy::loss_every)
      .def_readwrite("output_scale", &lc::AttentionIoPolicy::output_scale)
      .def_readwrite("output_bias", &lc::AttentionIoPolicy::output_bias)
      .def_readwrite("output_relu", &lc::AttentionIoPolicy::output_relu);

  py::class_<lc::AttentionSession>(m, "AttentionSession")
      .def(py::init([](std::size_t seq_len, std::size_t head_dim, bool causal, const std::string& device_name) {
             return lc::AttentionSession(lc::AttentionConfig{seq_len, head_dim, causal}, parseDevice(device_name));
           }),
           py::arg("seq_len"),
           py::arg("head_dim"),
           py::arg("causal") = false,
           py::arg("device") = "metal")
      .def("set_default_policy", &lc::AttentionSession::setDefaultPolicy)
      .def("forward",
           [](const lc::AttentionSession& s,
              const std::vector<float>& q,
              const std::vector<float>& k,
              const std::vector<float>& v) {
             if (q.size() != k.size() || q.size() != v.size()) {
               throw std::invalid_argument("q/k/v sizes must match");
             }
             std::vector<float> out(q.size(), 0.0f);
             throwIfNotSuccess(s.forward(q.data(), k.data(), v.data(), out.data()));
             return out;
           })
      .def("forward_with_policy",
           [](const lc::AttentionSession& s,
              const std::vector<float>& q,
              const std::vector<float>& k,
              const std::vector<float>& v,
              const lc::AttentionIoPolicy& policy) {
             if (q.size() != k.size() || q.size() != v.size()) {
               throw std::invalid_argument("q/k/v sizes must match");
             }
             std::vector<float> out(q.size(), 0.0f);
             throwIfNotSuccess(s.forwardWithPolicy(q.data(), k.data(), v.data(), out.data(), policy));
             return out;
           })
      .def("forward_into",
           [](const lc::AttentionSession& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& q,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& k,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& v,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             requireSize(k, static_cast<std::size_t>(q.size()), "k");
             requireSize(v, static_cast<std::size_t>(q.size()), "v");
             requireSize(out, static_cast<std::size_t>(q.size()), "out");
             throwIfNotSuccess(s.forward(q.data(), k.data(), v.data(), out.mutable_data()));
           })
      .def("forward_with_policy_into",
           [](const lc::AttentionSession& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& q,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& k,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& v,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out,
              const lc::AttentionIoPolicy& policy) {
             requireSize(k, static_cast<std::size_t>(q.size()), "k");
             requireSize(v, static_cast<std::size_t>(q.size()), "v");
             requireSize(out, static_cast<std::size_t>(q.size()), "out");
             throwIfNotSuccess(s.forwardWithPolicy(q.data(), k.data(), v.data(), out.mutable_data(), policy));
           })
      .def("train_step",
           [](const lc::AttentionSession& s,
              const std::vector<float>& q,
              const std::vector<float>& k,
              std::vector<float> v,
              const std::vector<float>& target,
              float learning_rate) {
             if (q.size() != k.size() || q.size() != v.size() || q.size() != target.size()) {
               throw std::invalid_argument("q/k/v/target sizes must match");
             }
             std::vector<float> out(q.size(), 0.0f);
             float loss = 0.0f;
             throwIfNotSuccess(s.trainStep(
                 q.data(), k.data(), v.data(), target.data(), out.data(), learning_rate, &loss));
             return py::make_tuple(out, v, loss);
           })
      .def("train_step_with_policy",
           [](const lc::AttentionSession& s,
              const std::vector<float>& q,
              const std::vector<float>& k,
              std::vector<float> v,
              const std::vector<float>& target,
              float learning_rate,
              const lc::AttentionIoPolicy& policy) {
             if (q.size() != k.size() || q.size() != v.size() || q.size() != target.size()) {
               throw std::invalid_argument("q/k/v/target sizes must match");
             }
             std::vector<float> out(q.size(), 0.0f);
             float loss = 0.0f;
             throwIfNotSuccess(s.trainStepWithPolicy(
                 q.data(), k.data(), v.data(), target.data(), out.data(), learning_rate, &loss, policy));
             return py::make_tuple(out, v, loss);
           });

  m.def("attention_forward",
        [](const std::vector<float>& q,
           const std::vector<float>& k,
           const std::vector<float>& v,
           std::size_t seq_len,
           std::size_t head_dim,
           bool causal,
           const std::string& device_name) {
          return attentionForwardVec(q, k, v, seq_len, head_dim, causal, parseDevice(device_name), nullptr);
        },
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("seq_len"),
        py::arg("head_dim"),
        py::arg("causal") = false,
        py::arg("device") = "metal");

  m.def("attention_forward",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& q,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& k,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& v,
           std::size_t seq_len,
           std::size_t head_dim,
           bool causal,
           const std::string& device_name) {
          const std::size_t expected = seq_len * head_dim;
          requireSize(q, expected, "q");
          requireSize(k, expected, "k");
          requireSize(v, expected, "v");

          py::array_t<float> out(expected);
          attentionForwardIntoCachedSession(
              q.data(), k.data(), v.data(), out.mutable_data(), seq_len, head_dim, causal, parseDevice(device_name), nullptr);
          return out;
        },
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("seq_len"),
        py::arg("head_dim"),
        py::arg("causal") = false,
        py::arg("device") = "metal");

  m.def("attention_forward_with_policy",
        [](const std::vector<float>& q,
           const std::vector<float>& k,
           const std::vector<float>& v,
           std::size_t seq_len,
           std::size_t head_dim,
           bool causal,
           const std::string& device_name,
           const lc::AttentionIoPolicy& policy) {
          return attentionForwardVec(q, k, v, seq_len, head_dim, causal, parseDevice(device_name), &policy);
        },
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("seq_len"),
        py::arg("head_dim"),
        py::arg("causal") = false,
        py::arg("device") = "metal",
        py::arg("policy") = lc::AttentionIoPolicy{});

  m.def("attention2d",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& q,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& k,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& v,
           bool causal,
           const std::string& device_name) {
          std::size_t seq_len = 0;
          std::size_t head_dim = 0;
          requireAttention2D(q.request(), "q", &seq_len, &head_dim);

          std::size_t k_seq = 0;
          std::size_t k_dim = 0;
          requireAttention2D(k.request(), "k", &k_seq, &k_dim);
          std::size_t v_seq = 0;
          std::size_t v_dim = 0;
          requireAttention2D(v.request(), "v", &v_seq, &v_dim);
          if (k_seq != seq_len || v_seq != seq_len || k_dim != head_dim || v_dim != head_dim) {
            throw std::invalid_argument("q/k/v must have identical 2D shape [seq_len, head_dim]");
          }

          py::array_t<float> out({static_cast<py::ssize_t>(seq_len), static_cast<py::ssize_t>(head_dim)});
          attentionForwardIntoCachedSession(
              q.data(), k.data(), v.data(), out.mutable_data(), seq_len, head_dim, causal, parseDevice(device_name), nullptr);
          return out;
        },
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("causal") = false,
        py::arg("device") = "metal");

  m.def("attention2d_into",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& q,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& k,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& v,
           py::array_t<float, py::array::c_style | py::array::forcecast>& out,
           bool causal,
           const std::string& device_name) {
          std::size_t seq_len = 0;
          std::size_t head_dim = 0;
          requireAttention2D(q.request(), "q", &seq_len, &head_dim);

          std::size_t k_seq = 0;
          std::size_t k_dim = 0;
          requireAttention2D(k.request(), "k", &k_seq, &k_dim);
          std::size_t v_seq = 0;
          std::size_t v_dim = 0;
          requireAttention2D(v.request(), "v", &v_seq, &v_dim);
          if (k_seq != seq_len || v_seq != seq_len || k_dim != head_dim || v_dim != head_dim) {
            throw std::invalid_argument("q/k/v must have identical 2D shape [seq_len, head_dim]");
          }

          auto out_info = out.request();
          if (out_info.ndim != 2 || out_info.shape.size() != 2 ||
              static_cast<std::size_t>(out_info.shape[0]) != seq_len ||
              static_cast<std::size_t>(out_info.shape[1]) != head_dim) {
            throw std::invalid_argument("out must be a 2D float32 array shaped [seq_len, head_dim]");
          }

          attentionForwardIntoCachedSession(
              q.data(), k.data(), v.data(), out.mutable_data(), seq_len, head_dim, causal, parseDevice(device_name), nullptr);
        },
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("out"),
        py::arg("causal") = false,
        py::arg("device") = "metal");

  m.def("attention2d_session",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& q,
           bool causal,
           const std::string& device_name) {
          std::size_t seq_len = 0;
          std::size_t head_dim = 0;
          requireAttention2D(q.request(), "q", &seq_len, &head_dim);
          return lc::AttentionSession(lc::AttentionConfig{seq_len, head_dim, causal}, parseDevice(device_name));
        },
        py::arg("q"),
        py::arg("causal") = false,
        py::arg("device") = "metal");
}
