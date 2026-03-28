#include "bind_common.hpp"

PYBIND11_MODULE(lightning_core, m) {
  m.doc() = "Lightning Core python bindings";

  bindRuntime(m);
  bindTensor(m);
  bindOps(m);
  bindAttention(m);
}
