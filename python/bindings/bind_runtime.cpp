#include "bind_common.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_map>

namespace {

lc::runtime::SyncMode parseSyncMode(const std::string& mode) {
  if (mode == "auto") {
    return lc::runtime::SyncMode::kAuto;
  }
  if (mode == "always") {
    return lc::runtime::SyncMode::kAlways;
  }
  if (mode == "never") {
    return lc::runtime::SyncMode::kNever;
  }
  throw std::invalid_argument("sync mode must be 'auto', 'always', or 'never'");
}

py::dict toBackendCapabilitiesDict(const lc::runtime::BackendCapabilities& caps) {
  py::dict out;
  out["device"] = toString(caps.device);
  out["built"] = caps.built;
  out["available"] = caps.available;
  out["compute_surface"] = caps.compute_surface;
  out["memory_surface"] = caps.memory_surface;
  out["sync_surface"] = caps.sync_surface;
  out["profiling_surface"] = caps.profiling_surface;
  out["runtime_trace_surface"] = caps.runtime_trace_surface;
  out["sync_policy_surface"] = caps.sync_policy_surface;
  out["memory_model"] = lc::runtime::memoryModelName(caps.memory_model);
  return out;
}

struct TimelineEventRow {
  std::size_t index{0};
  std::string type;
  std::string status;
  std::uint64_t timestamp_ns{0};
  std::uint64_t offset_ns{0};
  std::uint64_t delta_prev_ns{0};
  std::uint64_t delta_next_ns{0};
  std::size_t size_bytes{0};
  int detail0{0};
  int detail1{0};
};

struct TimelineGroupRow {
  std::string key;
  std::size_t count{0};
  std::uint64_t total_delta_next_ns{0};
  std::uint64_t max_delta_next_ns{0};
  std::size_t total_size_bytes{0};
  std::uint64_t first_timestamp_ns{0};
  std::uint64_t last_timestamp_ns{0};
};

template <typename T>
bool orderedLess(T lhs, T rhs, bool descending) {
  return descending ? lhs > rhs : lhs < rhs;
}

std::vector<TimelineEventRow> buildTimelineRows(const std::vector<lc::runtime::RuntimeTraceEvent>& events) {
  std::vector<TimelineEventRow> rows;
  rows.reserve(events.size());
  if (events.empty()) {
    return rows;
  }

  const std::uint64_t base_ts = events.front().timestamp_ns;
  for (std::size_t i = 0; i < events.size(); ++i) {
    const auto& ev = events[i];
    TimelineEventRow row;
    row.index = i;
    row.type = lc::runtime::runtimeTraceEventTypeName(ev.type);
    row.status = lc::runtime::getErrorString(ev.status);
    row.timestamp_ns = ev.timestamp_ns;
    row.offset_ns = (ev.timestamp_ns >= base_ts) ? (ev.timestamp_ns - base_ts) : 0;
    row.delta_prev_ns = (i > 0 && ev.timestamp_ns >= events[i - 1].timestamp_ns)
                            ? (ev.timestamp_ns - events[i - 1].timestamp_ns)
                            : 0;
    row.delta_next_ns =
        (i + 1 < events.size() && events[i + 1].timestamp_ns >= ev.timestamp_ns)
            ? (events[i + 1].timestamp_ns - ev.timestamp_ns)
            : 0;
    row.size_bytes = ev.size_bytes;
    row.detail0 = ev.detail0;
    row.detail1 = ev.detail1;
    rows.push_back(std::move(row));
  }
  return rows;
}

py::dict toTimelineEventDict(const TimelineEventRow& row) {
  py::dict out;
  out["index"] = row.index;
  out["type"] = row.type;
  out["status"] = row.status;
  out["timestamp_ns"] = row.timestamp_ns;
  out["offset_ns"] = row.offset_ns;
  out["delta_prev_ns"] = row.delta_prev_ns;
  out["delta_next_ns"] = row.delta_next_ns;
  out["size_bytes"] = row.size_bytes;
  out["detail0"] = row.detail0;
  out["detail1"] = row.detail1;
  return out;
}

std::string makeGroupKey(const TimelineEventRow& row, const std::string& group_by) {
  if (group_by == "type") {
    return row.type;
  }
  if (group_by == "status") {
    return row.status;
  }
  if (group_by == "type_status") {
    return row.type + "|" + row.status;
  }
  throw std::invalid_argument("group_by must be 'type', 'status', or 'type_status'");
}

void sortTimelineEvents(std::vector<TimelineEventRow>* rows,
                        const std::string& sort_by,
                        bool descending) {
  if (rows == nullptr) {
    return;
  }
  auto cmp = [&](const TimelineEventRow& a, const TimelineEventRow& b) {
    auto tie_break = [&]() {
      return orderedLess(a.index, b.index, descending);
    };

    if (sort_by == "index") {
      return orderedLess(a.index, b.index, descending);
    }
    if (sort_by == "timestamp_ns") {
      if (a.timestamp_ns != b.timestamp_ns) {
        return orderedLess(a.timestamp_ns, b.timestamp_ns, descending);
      }
      return tie_break();
    }
    if (sort_by == "offset_ns") {
      if (a.offset_ns != b.offset_ns) {
        return orderedLess(a.offset_ns, b.offset_ns, descending);
      }
      return tie_break();
    }
    if (sort_by == "delta_prev_ns") {
      if (a.delta_prev_ns != b.delta_prev_ns) {
        return orderedLess(a.delta_prev_ns, b.delta_prev_ns, descending);
      }
      return tie_break();
    }
    if (sort_by == "delta_next_ns") {
      if (a.delta_next_ns != b.delta_next_ns) {
        return orderedLess(a.delta_next_ns, b.delta_next_ns, descending);
      }
      return tie_break();
    }
    if (sort_by == "size_bytes") {
      if (a.size_bytes != b.size_bytes) {
        return orderedLess(a.size_bytes, b.size_bytes, descending);
      }
      return tie_break();
    }
    throw std::invalid_argument(
        "event_sort_by must be one of: index, timestamp_ns, offset_ns, delta_prev_ns, delta_next_ns, size_bytes");
  };
  std::stable_sort(rows->begin(), rows->end(), cmp);
}

void sortTimelineGroups(std::vector<TimelineGroupRow>* rows,
                        const std::string& sort_by,
                        bool descending) {
  if (rows == nullptr) {
    return;
  }
  auto cmp = [&](const TimelineGroupRow& a, const TimelineGroupRow& b) {
    auto tie_break = [&]() {
      return orderedLess(a.key, b.key, descending);
    };

    if (sort_by == "key") {
      return orderedLess(a.key, b.key, descending);
    }
    if (sort_by == "count") {
      if (a.count != b.count) {
        return orderedLess(a.count, b.count, descending);
      }
      return tie_break();
    }
    if (sort_by == "total_delta_next_ns") {
      if (a.total_delta_next_ns != b.total_delta_next_ns) {
        return orderedLess(a.total_delta_next_ns, b.total_delta_next_ns, descending);
      }
      return tie_break();
    }
    if (sort_by == "avg_delta_next_ns") {
      const std::uint64_t lhs_avg = (a.count == 0) ? 0 : (a.total_delta_next_ns / a.count);
      const std::uint64_t rhs_avg = (b.count == 0) ? 0 : (b.total_delta_next_ns / b.count);
      if (lhs_avg != rhs_avg) {
        return orderedLess(lhs_avg, rhs_avg, descending);
      }
      return tie_break();
    }
    if (sort_by == "max_delta_next_ns") {
      if (a.max_delta_next_ns != b.max_delta_next_ns) {
        return orderedLess(a.max_delta_next_ns, b.max_delta_next_ns, descending);
      }
      return tie_break();
    }
    if (sort_by == "total_size_bytes") {
      if (a.total_size_bytes != b.total_size_bytes) {
        return orderedLess(a.total_size_bytes, b.total_size_bytes, descending);
      }
      return tie_break();
    }
    throw std::invalid_argument(
        "group_sort_by must be one of: key, count, total_delta_next_ns, avg_delta_next_ns, max_delta_next_ns, "
        "total_size_bytes");
  };
  std::stable_sort(rows->begin(), rows->end(), cmp);
}

std::vector<TimelineGroupRow> buildTimelineGroups(const std::vector<TimelineEventRow>& rows,
                                                  const std::string& group_by) {
  std::vector<TimelineGroupRow> groups;
  std::unordered_map<std::string, std::size_t> group_index;
  groups.reserve(rows.size());
  group_index.reserve(rows.size());

  for (const auto& row : rows) {
    const std::string key = makeGroupKey(row, group_by);
    auto it = group_index.find(key);
    if (it == group_index.end()) {
      group_index.emplace(key, groups.size());
      TimelineGroupRow g;
      g.key = key;
      g.count = 0;
      g.total_delta_next_ns = 0;
      g.max_delta_next_ns = 0;
      g.total_size_bytes = 0;
      g.first_timestamp_ns = row.timestamp_ns;
      g.last_timestamp_ns = row.timestamp_ns;
      groups.push_back(std::move(g));
      it = group_index.find(key);
    }

    TimelineGroupRow& g = groups[it->second];
    if (g.count == 0) {
      g.first_timestamp_ns = row.timestamp_ns;
    }
    g.last_timestamp_ns = row.timestamp_ns;
    g.count += 1;
    g.total_delta_next_ns += row.delta_next_ns;
    g.max_delta_next_ns = std::max(g.max_delta_next_ns, row.delta_next_ns);
    g.total_size_bytes += row.size_bytes;
  }
  return groups;
}

py::dict toTimelineGroupDict(const TimelineGroupRow& row, std::uint64_t total_window_ns) {
  py::dict out;
  const double avg_delta_next_ns = (row.count == 0) ? 0.0 : static_cast<double>(row.total_delta_next_ns) /
                                                          static_cast<double>(row.count);
  const double window_share_pct = (total_window_ns == 0)
                                      ? 0.0
                                      : 100.0 * static_cast<double>(row.total_delta_next_ns) /
                                            static_cast<double>(total_window_ns);
  out["key"] = row.key;
  out["count"] = row.count;
  out["total_delta_next_ns"] = row.total_delta_next_ns;
  out["avg_delta_next_ns"] = avg_delta_next_ns;
  out["max_delta_next_ns"] = row.max_delta_next_ns;
  out["total_size_bytes"] = row.total_size_bytes;
  out["first_timestamp_ns"] = row.first_timestamp_ns;
  out["last_timestamp_ns"] = row.last_timestamp_ns;
  out["window_share_pct"] = window_share_pct;
  return out;
}

}  // namespace

void bindRuntime(py::module_& m) {
  m.def("cuda_available", [] { return lc::runtime::isCudaAvailable(); });
  m.def("metal_available", [] { return lc::runtime::isMetalAvailable(); });
  m.def("backend_name", [] { return lc::runtime::backendName(); });
  m.def("memory_model_name",
        [] { return std::string(lc::runtime::memoryModelName(lc::runtime::deviceMemoryModel())); });
  m.def("runtime_sync_policy_set",
        [](const std::string& mode, bool trace_sync_boundary) {
          lc::runtime::SyncPolicy policy;
          policy.mode = parseSyncMode(mode);
          policy.trace_sync_boundary = trace_sync_boundary;
          lc::runtime::setDefaultSyncPolicy(policy);
        },
        py::arg("mode") = "auto",
        py::arg("trace_sync_boundary") = false);
  m.def("runtime_sync_policy_get", [] {
    const auto policy = lc::runtime::defaultSyncPolicy();
    py::dict out;
    out["mode"] = lc::runtime::syncModeName(policy.mode);
    out["trace_sync_boundary"] = policy.trace_sync_boundary;
    return out;
  });
  m.def("runtime_sync_apply",
        [](const std::string& mode, bool trace_sync_boundary) {
          lc::runtime::SyncPolicy policy;
          policy.mode = parseSyncMode(mode);
          policy.trace_sync_boundary = trace_sync_boundary;
          throwIfNotSuccess(lc::runtime::deviceSynchronizeWithPolicy(policy));
        },
        py::arg("mode") = "auto",
        py::arg("trace_sync_boundary") = false);
  m.def("runtime_sync_apply_default", [] { throwIfNotSuccess(lc::runtime::applyDefaultSyncPolicy()); });
  m.def("runtime_backend_capabilities",
        [](const std::string& device) {
          return toBackendCapabilitiesDict(lc::runtime::backendCapabilities(parseDevice(device)));
        },
        py::arg("device"));
  m.def("runtime_active_backend_capabilities",
        [] { return toBackendCapabilitiesDict(lc::runtime::activeBackendCapabilities()); });
  m.def("runtime_trace_enable", [](bool enabled) { lc::runtime::setRuntimeTraceEnabled(enabled); }, py::arg("enabled"));
  m.def("runtime_trace_enabled", [] { return lc::runtime::isRuntimeTraceEnabled(); });
  m.def("runtime_trace_clear", [] { lc::runtime::clearRuntimeTraceEvents(); });
  m.def("runtime_trace_capacity", [] { return lc::runtime::runtimeTraceEventCapacity(); });
  m.def("runtime_trace_events", [] {
    py::list out;
    const auto events = lc::runtime::runtimeTraceEvents();
    for (const auto& ev : events) {
      py::dict row;
      row["type"] = lc::runtime::runtimeTraceEventTypeName(ev.type);
      row["status"] = lc::runtime::getErrorString(ev.status);
      row["timestamp_ns"] = ev.timestamp_ns;
      row["size_bytes"] = ev.size_bytes;
      row["detail0"] = ev.detail0;
      row["detail1"] = ev.detail1;
      out.append(row);
    }
    return out;
  });
  m.def("runtime_trace_timeline",
        [](const std::string& event_sort_by,
           bool event_descending,
           const std::string& group_by,
           const std::string& group_sort_by,
           bool group_descending,
           std::size_t hotspot_top_k) {
          const auto raw_events = lc::runtime::runtimeTraceEvents();
          const auto timeline_rows = buildTimelineRows(raw_events);

          py::dict out;
          out["event_count"] = timeline_rows.size();
          out["event_sort_by"] = event_sort_by;
          out["event_descending"] = event_descending;
          out["group_by"] = group_by;
          out["group_sort_by"] = group_sort_by;
          out["group_descending"] = group_descending;
          out["hotspot_top_k"] = hotspot_top_k;
          out["notes"] = "delta_next_ns is elapsed time until the next runtime trace event.";

          if (timeline_rows.empty()) {
            out["window_ns"] = static_cast<std::uint64_t>(0);
            out["events"] = py::list();
            out["groups"] = py::list();
            out["hotspots"] = py::list();
            return out;
          }

          const std::uint64_t window_ns =
              (timeline_rows.back().timestamp_ns >= timeline_rows.front().timestamp_ns)
                  ? (timeline_rows.back().timestamp_ns - timeline_rows.front().timestamp_ns)
                  : 0;
          out["window_ns"] = window_ns;

          std::vector<TimelineEventRow> sorted_events = timeline_rows;
          sortTimelineEvents(&sorted_events, event_sort_by, event_descending);
          py::list events_out;
          for (const auto& row : sorted_events) {
            events_out.append(toTimelineEventDict(row));
          }
          out["events"] = events_out;

          std::vector<TimelineGroupRow> groups = buildTimelineGroups(timeline_rows, group_by);
          sortTimelineGroups(&groups, group_sort_by, group_descending);
          py::list groups_out;
          for (const auto& g : groups) {
            groups_out.append(toTimelineGroupDict(g, window_ns));
          }
          out["groups"] = groups_out;

          std::vector<TimelineEventRow> hotspots = timeline_rows;
          sortTimelineEvents(&hotspots, "delta_next_ns", true);
          py::list hotspots_out;
          const std::size_t limit = (hotspot_top_k == 0) ? hotspots.size() : std::min(hotspot_top_k, hotspots.size());
          for (std::size_t i = 0; i < limit; ++i) {
            hotspots_out.append(toTimelineEventDict(hotspots[i]));
          }
          out["hotspots"] = hotspots_out;
          return out;
        },
        py::arg("event_sort_by") = "timestamp_ns",
        py::arg("event_descending") = false,
        py::arg("group_by") = "type",
        py::arg("group_sort_by") = "total_delta_next_ns",
        py::arg("group_descending") = true,
        py::arg("hotspot_top_k") = 8);
}
