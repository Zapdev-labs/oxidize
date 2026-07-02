#include "oxidize/prefetcher.hpp"

#include <cerrno>
#include <cstdio>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace oxidize {

LayerPrefetcher::LayerPrefetcher(const LayerRanges& ranges,
                                 const std::vector<int>& shard_fds,
                                 int lookahead)
    : ranges_(ranges), shard_fds_(shard_fds), lookahead_(lookahead) {
  if (lookahead_ > 0 && !ranges_.empty()) {
    size_t max_layer = ranges_.rbegin()->first;
    requested_.assign(max_layer + 1, false);
    thread_ = std::thread(&LayerPrefetcher::worker_loop, this);
  }
}

LayerPrefetcher::~LayerPrefetcher() {
  if (thread_.joinable()) {
    stop_.store(true, std::memory_order_release);
    cv_.notify_all();
    thread_.join();
  }
}

void LayerPrefetcher::request(size_t layer_index) {
  if (lookahead_ <= 0 || ranges_.empty()) return;
  if (layer_index >= requested_.size()) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (requested_[layer_index]) return;
    requested_[layer_index] = true;
    queue_.push(layer_index);
  }
  cv_.notify_one();
}

void LayerPrefetcher::prefetch_sync(size_t layer_index) {
  if (lookahead_ <= 0 || ranges_.empty()) return;
  if (layer_index >= requested_.size()) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!requested_[layer_index]) {
      requested_[layer_index] = true;
    }
  }
  issue_prefetch(layer_index);
}

void LayerPrefetcher::worker_loop() {
  while (!stop_.load(std::memory_order_acquire)) {
    size_t layer = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] {
        return stop_.load(std::memory_order_acquire) || !queue_.empty();
      });
      if (stop_.load(std::memory_order_acquire)) break;
      layer = queue_.front();
      queue_.pop();
    }
    issue_prefetch(layer);
  }
}

void LayerPrefetcher::issue_prefetch(size_t layer_index) {
  auto it = ranges_.find(layer_index);
  if (it == ranges_.end()) return;
  for (const ShardRange& r : it->second) {
    if (r.shard_index >= shard_fds_.size()) continue;
    int fd = shard_fds_[r.shard_index];
    if (fd < 0) continue;
#if defined(__linux__)
    // readahead is a hint; ignore errors (e.g. fs does not implement it).
    ssize_t rc = ::readahead(fd, static_cast<off_t>(r.offset),
                             static_cast<size_t>(r.length));
    if (rc < 0 && errno != EINVAL && errno != ESPIPE && errno != EOPNOTSUPP) {
      // Log once per layer at most. In practice this should be silent.
      std::fprintf(stderr,
                   "warning: readahead failed for layer %zu shard %zu: %d\n",
                   layer_index, r.shard_index, errno);
    }
#else
    (void)r;
#endif
  }
}

}  // namespace oxidize
