#include <algorithm>
#ifdef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../include/rbt.hpp"

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

struct BenchmarkReport {
  std::vector<std::chrono::steady_clock::duration> samples;
  std::size_t alloc_count = 0;
  std::size_t alloc_bytes = 0;
  bool hulls_match = false;
};

struct Summary {
  double total;
  double mean;
  double sd;
  double min;
  double median;
  double max;
  double p75;
  double p95;
  double p99;
};

double nearest_rank(const std::vector<double> &sorted, double percentile) {
  const auto rank = static_cast<std::size_t>(
      std::ceil(percentile * static_cast<double>(sorted.size())));
  return sorted[std::max<std::size_t>(1, rank) - 1];
}

Summary summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const double total =
      std::accumulate(samples.begin(), samples.end(), 0.0);
  const double mean = total / static_cast<double>(samples.size());
  double squared_deviation = 0.0;
  for (const double sample : samples)
    squared_deviation += (sample - mean) * (sample - mean);
  const double median = samples.size() % 2 == 0
                            ? (samples[samples.size() / 2 - 1] +
                               samples[samples.size() / 2]) /
                                  2.0
                            : samples[samples.size() / 2];
  return {total,
          mean,
          samples.size() > 1
              ? std::sqrt(squared_deviation /
                          static_cast<double>(samples.size() - 1))
              : 0.0,
          samples.front(),
          median,
          samples.back(),
          nearest_rank(samples, 0.75),
          nearest_rank(samples, 0.95),
          nearest_rank(samples, 0.99)};
}

struct AllocationSnapshot {
  std::size_t count;
  std::size_t bytes;
};

AllocationSnapshot capture_alloc() { return {g_alloc_count, g_alloc_bytes}; }

AllocationSnapshot delta_alloc(AllocationSnapshot before,
                               AllocationSnapshot after) {
  return {after.count - before.count, after.bytes - before.bytes};
}

std::vector<Point> baseline_hull(std::vector<Point> points,
                                 bool include_collinear) {
  if (points.empty())
    return {};
  if (points.size() == 1)
    return points;

  Point pivot = *std::min_element(
      points.begin(), points.end(), [](const Point &a, const Point &b) {
        return std::tie(a.y, a.x) < std::tie(b.y, b.x);
      });

  std::sort(points.begin(), points.end(), [&](const Point &a, const Point &b) {
    if (a == pivot)
      return true;
    if (b == pivot)
      return false;

    long long dx_a = a.x - pivot.x, dy_a = a.y - pivot.y;
    long long dx_b = b.x - pivot.x, dy_b = b.y - pivot.y;

    bool a_upper = dy_a > 0 || (dy_a == 0 && dx_a >= 0);
    bool b_upper = dy_b > 0 || (dy_b == 0 && dx_b >= 0);
    if (a_upper != b_upper)
      return a_upper;

    __int128 cross_prod =
        static_cast<__int128>(dx_a) * dy_b - static_cast<__int128>(dy_a) * dx_b;
    if (cross_prod != 0)
      return cross_prod > 0;

    __int128 dist_a =
        static_cast<__int128>(dx_a) * dx_a + static_cast<__int128>(dy_a) * dy_a;
    __int128 dist_b =
        static_cast<__int128>(dx_b) * dx_b + static_cast<__int128>(dy_b) * dy_b;
    return dist_a < dist_b;
  });

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

  return hull;
}

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
      report.hulls_match = (tree_hull == base_hull);
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
      report.hulls_match = (tree_hull == base_hull);
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
      report.hulls_match = (tree_hull == base_hull);
    }
  }

  return report;
}

void print_report(const char *label, BenchmarkReport report) {
  std::sort(report.samples.begin(), report.samples.end());
  const auto micros = [](auto duration) {
    return std::chrono::duration_cast<std::chrono::microseconds>(duration)
        .count();
  };
  printf("%-25s | %9lld | %11lld | %8lld  | %10s | %7zu | %5zu\n", label,
         micros(report.samples.front()),
         micros(report.samples[report.samples.size() / 2]),
         micros(report.samples.back()), report.hulls_match ? "yes" : "no",
         report.alloc_count, report.alloc_bytes);
}

void expect(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void test_statistics() {
  const auto summary = summarize({1.0, 2.0, 3.0, 4.0});
  expect(summary.total == 10.0, "statistics total");
  expect(summary.mean == 2.5, "statistics mean");
  expect(std::abs(summary.sd - std::sqrt(5.0 / 3.0)) < 1e-12,
         "statistics sample standard deviation");
  expect(summary.min == 1.0, "statistics minimum");
  expect(summary.median == 2.5, "statistics median");
  expect(summary.max == 4.0, "statistics maximum");
  expect(summary.p75 == 3.0, "statistics p75");
  expect(summary.p95 == 4.0, "statistics p95");
  expect(summary.p99 == 4.0, "statistics p99");
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
    test_statistics();
    DynamicHull tree;
    assert(tree.insert({0, 0}));
    assert(tree.insert({2, 0}));
    assert(tree.insert({1, 1}));
    assert(!tree.insert({1, 1}));
    assert(tree.size() == 3);
    assert(tree.valid());
    const auto points = tree.ordered_points();
    const std::vector<Point> expected{{0, 0}, {2, 0}, {1, 1}};
    assert(points == expected);

    assert(tree.erase({2, 0}));
    assert(!tree.erase({2, 0}));
    assert(tree.valid());
    assert(tree.insert({-1, -1}));
    assert(tree.valid());
    const auto size_before_pivot_erase = tree.size();
    assert(tree.erase({-1, -1}));
    assert(tree.size() == size_before_pivot_erase - 1);
    const auto points_after_pivot_erase = tree.ordered_points();
    assert(std::find(points_after_pivot_erase.begin(),
                     points_after_pivot_erase.end(),
                     Point{-1, -1}) == points_after_pivot_erase.end());
    assert(tree.valid());

    DynamicHull hull_tree;
    const std::vector<Point> test_points{{0, 0}, {1, 0}, {2, 0},
                                         {2, 2}, {0, 2}, {1, 1}};
    for (const auto &p : test_points) {
      hull_tree.insert(p);
    }

    const auto tree_hull_excl = hull_tree.hull(false);
    const auto base_hull_excl = baseline_hull(test_points, false);
    assert(tree_hull_excl == base_hull_excl);

    const auto tree_hull_incl = hull_tree.hull(true);
    const auto base_hull_incl = baseline_hull(test_points, true);
    assert(tree_hull_incl == base_hull_incl);

    const std::vector<Point> smoke_points{{0, 0}, {1, 0}, {2, 0}};
    const std::vector<Point> smoke_candidates{{1, 1}, {0, 1}};
    const auto report =
        benchmark_normal_insert(smoke_points, smoke_candidates, 1);
    assert(report.samples.size() == 1);
    assert(report.hulls_match);
  }

  if (run_benchmark) {
    std::vector<Point> points;
    std::vector<Point> candidates;

    for (int i = 0; i < 100; ++i) {
      points.push_back({i, (i * 37 + 13) % 128});
    }

    for (int i = 0; i < 50; ++i) {
      candidates.push_back({i * 2, (i * 61 + 29) % 256});
    }

    const int num_samples = 31;

    printf("Scenario                  | Min (us)  | Median (us) | Max (us)  | "
           "Hulls Match | Allocs | Bytes\n");
    printf(
        "--------------------------------------------------------------------"
        "--------\n");

    benchmark_baseline(points, 1);
    print_report("Baseline (std::sort)",
                 benchmark_baseline(points, num_samples));

    benchmark_normal_insert(points, candidates, 1);
    print_report("Normal insert + hull",
                 benchmark_normal_insert(points, candidates, num_samples));

    benchmark_normal_delete(points, candidates, 1);
    print_report("Normal delete + hull",
                 benchmark_normal_delete(points, candidates, num_samples));

    benchmark_normal_insert(points, candidates, 1);
    print_report("Pivot-changing insert",
                 benchmark_normal_insert(points, candidates, num_samples));

    benchmark_pivot_delete(points, 1);
    print_report("Pivot deletion + rebuild",
                 benchmark_pivot_delete(points, num_samples));
  }

  return 0;
}
