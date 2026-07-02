#pragma once
// Async layer-ahead prefetch for mmap'd GGUF weights.
//
// Uses Linux readahead(2) to tell the kernel to fetch the pages for an upcoming
// layer while the current layer is being computed. This hides SSD/page-cache
// latency when the working set is larger than RAM.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "oxidize/gguf.hpp"

namespace oxidize {

class LayerPrefetcher {
 public:
  // `shard_fds` must stay valid for the lifetime of the prefetcher. A negative
  // fd for a shard means "no readahead for that shard".
  LayerPrefetcher(const LayerRanges& ranges,
                  const std::vector<int>& shard_fds,
                  int lookahead);
  ~LayerPrefetcher();

  // Request async prefetch of `layer_index`. Repeated requests for the same
  // layer are ignored. Requests for unknown layers are ignored.
  void request(size_t layer_index);

  // Issue readahead hints synchronously for `layer_index`. Useful for warming
  // the first layer(s) before the compute loop starts.
  void prefetch_sync(size_t layer_index);

  LayerPrefetcher(const LayerPrefetcher&) = delete;
  LayerPrefetcher& operator=(const LayerPrefetcher&) = delete;

 private:
  void worker_loop();
  void issue_prefetch(size_t layer_index);

  const LayerRanges& ranges_;
  std::vector<int> shard_fds_;
  int lookahead_ = 0;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<size_t> queue_;
  std::vector<bool> requested_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

}  // namespace oxidize
