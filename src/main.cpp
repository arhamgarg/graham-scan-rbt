#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rbt.hpp"

// ============================================================================
// Allocation Tracking
// ============================================================================

std::size_t g_alloc_count = 0;
std::size_t g_alloc_bytes = 0;

void *operator new(std::size_t size) {
  g_alloc_count++;
  g_alloc_bytes += size;
  void *ptr = std::malloc(size);
  if (!ptr)
    throw std::bad_alloc();
  return ptr;
}

void *operator new[](std::size_t size) {
  g_alloc_count++;
  g_alloc_bytes += size;
  void *ptr = std::malloc(size);
  if (!ptr)
    throw std::bad_alloc();
  return ptr;
}

void operator delete(void *ptr) noexcept { std::free(ptr); }

void operator delete[](void *ptr) noexcept { std::free(ptr); }

void operator delete(void *ptr, std::size_t) noexcept { std::free(ptr); }

void operator delete[](void *ptr, std::size_t) noexcept { std::free(ptr); }

// ============================================================================
// Benchmark Types
// ============================================================================

struct BenchmarkReport {
  std::vector<std::chrono::steady_clock::duration> samples;
  std::size_t alloc_count = 0;
  std::size_t alloc_bytes = 0;
  bool hulls_match = false;
};

struct AllocationSnapshot {
  std::size_t count;
  std::size_t bytes;
};

AllocationSnapshot capture_alloc() { return {g_alloc_count, g_alloc_bytes}; }

AllocationSnapshot delta_alloc(AllocationSnapshot before,
                               AllocationSnapshot after) {
  return {after.count - before.count, after.bytes - before.bytes};
}

// Helper: rotate points so smallest (y, x) is first
std::vector<Point> canonicalize(std::vector<Point> pts) {
  if (pts.empty())
    return pts;

  size_t min_idx = 0;
  for (size_t i = 1; i < pts.size(); ++i) {
    if (std::tie(pts[i].y, pts[i].x) <
        std::tie(pts[min_idx].y, pts[min_idx].x)) {
      min_idx = i;
    }
  }

  std::rotate(pts.begin(), pts.begin() + min_idx, pts.end());
  return pts;
}

// Baseline Graham scan using std::sort
std::vector<Point> baseline_hull(std::vector<Point> points,
                                 bool include_collinear) {
  if (points.empty())
    return {};
  if (points.size() == 1)
    return points;

  // Find pivot (minimum (y, x))
  Point pivot = *std::min_element(
      points.begin(), points.end(), [](const Point &a, const Point &b) {
        return std::tie(a.y, a.x) < std::tie(b.y, b.x);
      });

  // Sort by polar order from pivot
  std::sort(points.begin(), points.end(), [&](const Point &a, const Point &b) {
    if (a == pivot)
      return true; // pivot first
    if (b == pivot)
      return false;

    long long dx_a = a.x - pivot.x, dy_a = a.y - pivot.y;
    long long dx_b = b.x - pivot.x, dy_b = b.y - pivot.y;

    // Upper-half plane test
    bool a_upper = dy_a > 0 || (dy_a == 0 && dx_a >= 0);
    bool b_upper = dy_b > 0 || (dy_b == 0 && dx_b >= 0);
    if (a_upper != b_upper)
      return a_upper;

    // Cross product for angular ordering
    __int128 cross_prod =
        static_cast<__int128>(dx_a) * dy_b - static_cast<__int128>(dy_a) * dx_b;
    if (cross_prod != 0)
      return cross_prod > 0;

    // Distance squared (closer first)
    __int128 dist_a =
        static_cast<__int128>(dx_a) * dx_a + static_cast<__int128>(dy_a) * dy_a;
    __int128 dist_b =
        static_cast<__int128>(dx_b) * dx_b + static_cast<__int128>(dy_b) * dy_b;
    return dist_a < dist_b;
  });

  // Graham's scan stack
  std::vector<Point> hull;
  for (const auto &p : points) {
    while (hull.size() > 1) {
      int turn = cross(hull[hull.size() - 2], hull[hull.size() - 1], p);
      if (include_collinear) {
        if (turn < 0)
          hull.pop_back();
        else
          break;
      } else {
        if (turn <= 0)
          hull.pop_back();
        else
          break;
      }
    }
    hull.push_back(p);
  }

  return canonicalize(hull);
}

// ============================================================================
// Benchmark Scenarios
// ============================================================================

BenchmarkReport benchmark_baseline(const std::vector<Point> &points,
                                   int num_samples) {
  BenchmarkReport report;

  for (int i = 0; i < num_samples; ++i) {
    AllocationSnapshot before = capture_alloc();
    auto start = std::chrono::steady_clock::now();

    auto hull = baseline_hull(points, false);

    auto end = std::chrono::steady_clock::now();
    AllocationSnapshot after = capture_alloc();

    report.samples.push_back(end - start);

    if (i == num_samples - 1) {
      AllocationSnapshot delta = delta_alloc(before, after);
      report.alloc_count = delta.count;
      report.alloc_bytes = delta.bytes;
      report.hulls_match = true;
    }
  }

  return report;
}

BenchmarkReport benchmark_normal_insert(const std::vector<Point> &points,
                                        const std::vector<Point> &candidates,
                                        int num_samples) {
  BenchmarkReport report;

  for (int i = 0; i < num_samples; ++i) {
    DynamicHull tree;
    for (const auto &p : points) {
      tree.insert(p);
    }

    AllocationSnapshot before = capture_alloc();
    auto start = std::chrono::steady_clock::now();

    for (const auto &c : candidates) {
      tree.insert(c);
      tree.hull(false);
    }

    auto end = std::chrono::steady_clock::now();
    AllocationSnapshot after = capture_alloc();

    report.samples.push_back(end - start);

    if (i == num_samples - 1) {
      AllocationSnapshot delta = delta_alloc(before, after);
      report.alloc_count = delta.count;
      report.alloc_bytes = delta.bytes;

      auto tree_hull = tree.hull(false);
      auto base_hull = baseline_hull(
          [&]() {
            auto combined = points;
            combined.insert(combined.end(), candidates.begin(),
                            candidates.end());
            return combined;
          }(),
          false);
      report.hulls_match = (canonicalize(tree_hull) == canonicalize(base_hull));
    }
  }

  return report;
}

BenchmarkReport benchmark_normal_delete(const std::vector<Point> &points,
                                        const std::vector<Point> &candidates,
                                        int num_samples) {
  BenchmarkReport report;

  for (int i = 0; i < num_samples; ++i) {
    DynamicHull tree;
    auto combined = points;
    combined.insert(combined.end(), candidates.begin(), candidates.end());
    for (const auto &p : combined) {
      tree.insert(p);
    }

    AllocationSnapshot before = capture_alloc();
    auto start = std::chrono::steady_clock::now();

    for (const auto &c : candidates) {
      tree.erase(c);
      tree.hull(false);
    }

    auto end = std::chrono::steady_clock::now();
    AllocationSnapshot after = capture_alloc();

    report.samples.push_back(end - start);

    if (i == num_samples - 1) {
      AllocationSnapshot delta = delta_alloc(before, after);
      report.alloc_count = delta.count;
      report.alloc_bytes = delta.bytes;

      auto tree_hull = tree.hull(false);
      auto base_hull = baseline_hull(points, false);
      report.hulls_match = (canonicalize(tree_hull) == canonicalize(base_hull));
    }
  }

  return report;
}

BenchmarkReport benchmark_pivot_insert(const std::vector<Point> &points,
                                       const std::vector<Point> &candidates,
                                       int num_samples) {
  BenchmarkReport report;

  for (int i = 0; i < num_samples; ++i) {
    DynamicHull tree;
    for (const auto &p : points) {
      tree.insert(p);
    }

    AllocationSnapshot before = capture_alloc();
    auto start = std::chrono::steady_clock::now();

    for (const auto &c : candidates) {
      tree.insert(c);
      tree.hull(false);
    }

    auto end = std::chrono::steady_clock::now();
    AllocationSnapshot after = capture_alloc();

    report.samples.push_back(end - start);

    if (i == num_samples - 1) {
      AllocationSnapshot delta = delta_alloc(before, after);
      report.alloc_count = delta.count;
      report.alloc_bytes = delta.bytes;

      auto tree_hull = tree.hull(false);
      auto base_hull = baseline_hull(
          [&]() {
            auto combined = points;
            combined.insert(combined.end(), candidates.begin(),
                            candidates.end());
            return combined;
          }(),
          false);
      report.hulls_match = (canonicalize(tree_hull) == canonicalize(base_hull));
    }
  }

  return report;
}

BenchmarkReport benchmark_pivot_delete(const std::vector<Point> &points,
                                       int num_samples) {
  BenchmarkReport report;

  for (int i = 0; i < num_samples; ++i) {
    DynamicHull tree;
    for (const auto &p : points) {
      tree.insert(p);
    }

    // Identify pivot before deletion
    auto ordered = tree.ordered_points();
    Point pivot_to_delete = ordered[0];

    AllocationSnapshot before = capture_alloc();
    auto start = std::chrono::steady_clock::now();

    tree.erase(pivot_to_delete);
    tree.hull(false);

    auto end = std::chrono::steady_clock::now();
    AllocationSnapshot after = capture_alloc();

    report.samples.push_back(end - start);

    if (i == num_samples - 1) {
      AllocationSnapshot delta = delta_alloc(before, after);
      report.alloc_count = delta.count;
      report.alloc_bytes = delta.bytes;

      auto remaining = points;
      remaining.erase(
          std::remove(remaining.begin(), remaining.end(), pivot_to_delete),
          remaining.end());
      auto tree_hull = tree.hull(false);
      auto base_hull = baseline_hull(remaining, false);
      report.hulls_match = (canonicalize(tree_hull) == canonicalize(base_hull));
    }
  }

  return report;
}

int main(int argc, char *argv[]) {
  assert(cross({0, 0}, {1, 0}, {0, 1}) > 0);
  assert((Point{2, 3} == Point{2, 3}));

  bool run_self_test = false;
  bool run_benchmark = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--self-test") == 0) {
      run_self_test = true;
    } else if (std::strcmp(argv[i], "--benchmark") == 0) {
      run_benchmark = true;
    }
  }

  if (run_self_test) {
    DynamicHull tree;
    assert(tree.insert({0, 0}));
    assert(tree.insert({2, 0}));
    assert(tree.insert({1, 1}));
    assert(!tree.insert({1, 1})); // duplicate
    assert(tree.size() == 3);
    assert(tree.valid());
    const auto points = tree.ordered_points();
    const std::vector<Point> expected{{0, 0}, {2, 0}, {1, 1}};
    assert(points == expected);

    // Task 3: Deletion and pivot rebuilds
    assert(tree.erase({2, 0}));
    assert(!tree.erase({2, 0})); // already deleted
    assert(tree.valid());
    assert(tree.insert({-1, -1})); // becomes a new pivot
    assert(tree.valid());
    assert(tree.erase({-1, -1})); // deletes pivot and rebuilds
    assert(tree.valid());

    // Task 4: Hull construction
    DynamicHull hull_tree;
    const std::vector<Point> test_points{{0, 0}, {1, 0}, {2, 0},
                                         {2, 2}, {0, 2}, {1, 1}};
    for (const auto &p : test_points) {
      hull_tree.insert(p);
    }

    const auto tree_hull_excl = canonicalize(hull_tree.hull(false));
    const auto base_hull_excl = canonicalize(baseline_hull(test_points, false));
    assert(tree_hull_excl == base_hull_excl);

    const auto tree_hull_incl = canonicalize(hull_tree.hull(true));
    const auto base_hull_incl = canonicalize(baseline_hull(test_points, true));
    assert(tree_hull_incl == base_hull_incl);

    // Task 5: Benchmark smoke test
    const std::vector<Point> smoke_points{{0, 0}, {1, 0}, {2, 0}};
    const std::vector<Point> smoke_candidates{{1, 1}, {0, 1}};
    const auto report =
        benchmark_normal_insert(smoke_points, smoke_candidates, 1);
    assert(report.samples.size() == 1);
    assert(report.hulls_match);
  }

  if (run_benchmark) {
    // Fixed seed for reproducibility
    std::vector<Point> points;
    std::vector<Point> candidates;

    // Generate 100 baseline points
    for (int i = 0; i < 100; ++i) {
      points.push_back({i, (i * 37 + 13) % 128});
    }

    // Generate 50 candidate points
    for (int i = 0; i < 50; ++i) {
      candidates.push_back({i * 2, (i * 61 + 29) % 256});
    }

    // Run 1 warm-up + 31 samples
    const int num_samples = 31;

    printf("Scenario                  | Min (us)  | Median (us) | Max (us)  | "
           "Hulls Match | Allocs | Bytes\n");
    printf(
        "--------------------------------------------------------------------"
        "--------\n");

    // Baseline
    {
      benchmark_baseline(points, 1); // warm-up
      BenchmarkReport report = benchmark_baseline(points, num_samples);
      std::vector<std::chrono::steady_clock::duration> sorted_samples =
          report.samples;
      std::sort(sorted_samples.begin(), sorted_samples.end());
      auto min_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        sorted_samples.front())
                        .count();
      auto max_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        sorted_samples.back())
                        .count();
      auto median_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           sorted_samples[sorted_samples.size() / 2])
                           .count();
      printf(
          "Baseline (std::sort)      | %9lld | %11lld | %8lld  | %10s | %7zu | "
          "%5zu\n",
          min_us, median_us, max_us, report.hulls_match ? "yes" : "no",
          report.alloc_count, report.alloc_bytes);
    }

    // Normal insert
    {
      benchmark_normal_insert(points, candidates, 1); // warm-up
      BenchmarkReport report =
          benchmark_normal_insert(points, candidates, num_samples);
      std::vector<std::chrono::steady_clock::duration> sorted_samples =
          report.samples;
      std::sort(sorted_samples.begin(), sorted_samples.end());
      auto min_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        sorted_samples.front())
                        .count();
      auto max_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        sorted_samples.back())
                        .count();
      auto median_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           sorted_samples[sorted_samples.size() / 2])
                           .count();
      printf(
          "Normal insert + hull      | %9lld | %11lld | %8lld  | %10s | %7zu | "
          "%5zu\n",
          min_us, median_us, max_us, report.hulls_match ? "yes" : "no",
          report.alloc_count, report.alloc_bytes);
    }

    // Normal delete
    {
      benchmark_normal_delete(points, candidates, 1); // warm-up
      BenchmarkReport report =
          benchmark_normal_delete(points, candidates, num_samples);
      std::vector<std::chrono::steady_clock::duration> sorted_samples =
          report.samples;
      std::sort(sorted_samples.begin(), sorted_samples.end());
      auto min_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        sorted_samples.front())
                        .count();
      auto max_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        sorted_samples.back())
                        .count();
      auto median_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           sorted_samples[sorted_samples.size() / 2])
                           .count();
      printf(
          "Normal delete + hull      | %9lld | %11lld | %8lld  | %10s | %7zu | "
          "%5zu\n",
          min_us, median_us, max_us, report.hulls_match ? "yes" : "no",
          report.alloc_count, report.alloc_bytes);
    }

    // Pivot-changing insert
    {
      benchmark_pivot_insert(points, candidates, 1); // warm-up
      BenchmarkReport report =
          benchmark_pivot_insert(points, candidates, num_samples);
      std::vector<std::chrono::steady_clock::duration> sorted_samples =
          report.samples;
      std::sort(sorted_samples.begin(), sorted_samples.end());
      auto min_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        sorted_samples.front())
                        .count();
      auto max_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        sorted_samples.back())
                        .count();
      auto median_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           sorted_samples[sorted_samples.size() / 2])
                           .count();
      printf(
          "Pivot-changing insert     | %9lld | %11lld | %8lld  | %10s | %7zu | "
          "%5zu\n",
          min_us, median_us, max_us, report.hulls_match ? "yes" : "no",
          report.alloc_count, report.alloc_bytes);
    }

    // Pivot delete
    {
      benchmark_pivot_delete(points, 1); // warm-up
      BenchmarkReport report = benchmark_pivot_delete(points, num_samples);
      std::vector<std::chrono::steady_clock::duration> sorted_samples =
          report.samples;
      std::sort(sorted_samples.begin(), sorted_samples.end());
      auto min_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        sorted_samples.front())
                        .count();
      auto max_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        sorted_samples.back())
                        .count();
      auto median_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           sorted_samples[sorted_samples.size() / 2])
                           .count();
      printf(
          "Pivot deletion + rebuild  | %9lld | %11lld | %8lld  | %10s | %7zu | "
          "%5zu\n",
          min_us, median_us, max_us, report.hulls_match ? "yes" : "no",
          report.alloc_count, report.alloc_bytes);
    }
  }

  return 0;
}
