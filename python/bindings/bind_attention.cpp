#include "bind_common.hpp"

namespace {

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

  lc::AttentionConfig cfg{seq_len, head_dim, causal};
  std::vector<float> out(expected, 0.0f);
  if (policy == nullptr) {
    throwIfNotSuccess(lc::attentionForward(q.data(), k.data(), v.data(), out.data(), cfg, device));
  } else {
    throwIfNotSuccess(lc::attentionForwardWithPolicy(q.data(), k.data(), v.data(), out.data(), cfg, device, *policy));
  }
  return out;
}

}  // namespace

void bindAttention(py::module_& m) {
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
          auto out = attentionForwardVec(toVector(q), toVector(k), toVector(v), seq_len, head_dim, causal, parseDevice(device_name), nullptr);
          return toNumpy(out);
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
}
