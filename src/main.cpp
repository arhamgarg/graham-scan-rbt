#include <algorithm>
#ifdef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
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
  const char *name;
  std::vector<double> samples;
  double allocations_per_operation = 0.0;
  double bytes_per_operation = 0.0;
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

struct BenchmarkConfig {
  std::size_t dataset_size;
  std::uint32_t seed;
  std::size_t warmups;
  std::size_t runs;
  std::size_t fast_batch_size;
};

struct Dataset {
  std::vector<Point> points;
  std::vector<std::pair<long long, long long>> coordinates;
  std::vector<Point> normal_inserts;
  std::vector<Point> normal_deletes;
  Point pivot;
  Point pivot_insert;
};

Dataset generate_dataset(const BenchmarkConfig &config) {
  std::mt19937 generator(config.seed);
  std::set<std::pair<long long, long long>> seen;
  Dataset dataset;
  dataset.points.reserve(config.dataset_size);

  while (dataset.points.size() < config.dataset_size) {
    const Point point{static_cast<std::int32_t>(generator()),
                      static_cast<std::int32_t>(generator())};
    if (seen.emplace(point.x, point.y).second)
      dataset.points.push_back(point);
  }

  dataset.coordinates.assign(seen.begin(), seen.end());
  dataset.pivot = *std::min_element(
      dataset.points.begin(), dataset.points.end(), [](Point a, Point b) {
        return std::tie(a.y, a.x) < std::tie(b.y, b.x);
      });
  dataset.pivot_insert = {dataset.pivot.x, dataset.pivot.y - 1};
  while (!seen.emplace(dataset.pivot_insert.x, dataset.pivot_insert.y).second)
    --dataset.pivot_insert.y;

  while (dataset.normal_inserts.size() < config.fast_batch_size) {
    const Point point{static_cast<std::int32_t>(generator()),
                      static_cast<std::int32_t>(generator())};
    if (std::tie(point.y, point.x) >
            std::tie(dataset.pivot.y, dataset.pivot.x) &&
        seen.emplace(point.x, point.y).second)
      dataset.normal_inserts.push_back(point);
  }

  for (const Point point : dataset.points) {
    if (!(point == dataset.pivot))
      dataset.normal_deletes.push_back(point);
    if (dataset.normal_deletes.size() == config.fast_batch_size)
      break;
  }
  return dataset;
}

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

volatile std::size_t g_benchmark_sink = 0;

void consume(std::size_t value) { g_benchmark_sink ^= value; }

std::unique_ptr<DynamicHull> make_tree(const std::vector<Point> &points) {
  auto tree = std::make_unique<DynamicHull>();
  for (const Point point : points)
    tree->insert(point);
  return tree;
}

template <typename Setup, typename Operation>
BenchmarkReport measure(const char *name, const BenchmarkConfig &config,
                        std::size_t operations_per_run, Setup setup,
                        Operation operation) {
  BenchmarkReport report{name, {}};
  report.samples.reserve(config.runs);

  for (std::size_t run = 0; run < config.warmups; ++run) {
    auto state = setup();
    consume(operation(state));
  }

  for (std::size_t run = 0; run < config.runs; ++run) {
    auto state = setup();
    const auto start = std::chrono::steady_clock::now();
    const auto result = operation(state);
    const auto end = std::chrono::steady_clock::now();
    report.samples.push_back(
        std::chrono::duration<double, std::nano>(end - start).count() /
        static_cast<double>(operations_per_run));
    consume(result);
  }

  auto state = setup();
  const auto before = capture_alloc();
  consume(operation(state));
  const auto allocation = delta_alloc(before, capture_alloc());
  report.allocations_per_operation =
      static_cast<double>(allocation.count) / operations_per_run;
  report.bytes_per_operation =
      static_cast<double>(allocation.bytes) / operations_per_run;
  return report;
}

struct HullState {
  std::unique_ptr<DynamicHull> tree;
  std::vector<Point> hull;
};

struct BuildState {
  const std::vector<Point> *points;
  std::unique_ptr<DynamicHull> tree;
  std::vector<Point> hull;
};

struct MutationState {
  std::unique_ptr<DynamicHull> tree;
  const std::vector<Point> *points;
};

std::vector<BenchmarkReport>
run_benchmarks(const Dataset &dataset, const BenchmarkConfig &config) {
  std::vector<BenchmarkReport> reports;
  reports.reserve(7);

  reports.push_back(measure(
      "Batch Graham scan", config, 1,
      [&] { return BuildState{&dataset.points, nullptr, {}}; },
      [](BuildState &state) {
        state.hull = baseline_hull(*state.points, false);
        return state.hull.size();
      }));

  reports.push_back(measure(
      "RBT build + hull", config, 1,
      [&] { return BuildState{&dataset.points, nullptr, {}}; },
      [](BuildState &state) {
        state.tree = make_tree(*state.points);
        state.hull = state.tree->hull(false);
        return state.hull.size();
      }));

  reports.push_back(measure(
      "RBT hull query", config, 1,
      [&] { return HullState{make_tree(dataset.points), {}}; },
      [](HullState &state) {
        state.hull = state.tree->hull(false);
        return state.hull.size();
      }));

  reports.push_back(measure(
      "RBT normal insert", config, config.fast_batch_size,
      [&] {
        return MutationState{make_tree(dataset.points),
                             &dataset.normal_inserts};
      },
      [](MutationState &state) {
        for (const Point point : *state.points)
          state.tree->insert(point);
        return state.tree->size();
      }));

  reports.push_back(measure(
      "RBT normal delete", config, config.fast_batch_size,
      [&] {
        return MutationState{make_tree(dataset.points),
                             &dataset.normal_deletes};
      },
      [](MutationState &state) {
        for (const Point point : *state.points)
          state.tree->erase(point);
        return state.tree->size();
      }));

  reports.push_back(measure(
      "RBT pivot insert", config, 1,
      [&] { return MutationState{make_tree(dataset.points), nullptr}; },
      [&](MutationState &state) {
        state.tree->insert(dataset.pivot_insert);
        return state.tree->size();
      }));

  reports.push_back(measure(
      "RBT pivot delete", config, 1,
      [&] { return MutationState{make_tree(dataset.points), nullptr}; },
      [&](MutationState &state) {
        state.tree->erase(dataset.pivot);
        return state.tree->size();
      }));
  return reports;
}

void expect(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void verify_workloads(const Dataset &dataset) {
  const auto expected = baseline_hull(dataset.points, false);
  auto tree = make_tree(dataset.points);
  expect(tree->valid(), "RBT build invariants");
  expect(tree->hull(false) == expected, "RBT build hull");

  const auto original_size = tree->size();
  for (const Point point : dataset.normal_inserts)
    expect(tree->insert(point), "normal insert accepted");
  auto inserted_points = dataset.points;
  inserted_points.insert(inserted_points.end(), dataset.normal_inserts.begin(),
                         dataset.normal_inserts.end());
  expect(tree->size() == original_size + dataset.normal_inserts.size(),
         "normal insert size");
  expect(tree->valid(), "normal insert invariants");
  expect(tree->hull(false) == baseline_hull(inserted_points, false),
         "normal insert hull");

  tree = make_tree(dataset.points);
  for (const Point point : dataset.normal_deletes)
    expect(tree->erase(point), "normal delete accepted");
  auto remaining = dataset.points;
  for (const Point point : dataset.normal_deletes)
    remaining.erase(std::remove(remaining.begin(), remaining.end(), point),
                    remaining.end());
  expect(tree->size() == original_size - dataset.normal_deletes.size(),
         "normal delete size");
  expect(tree->valid(), "normal delete invariants");
  expect(tree->hull(false) == baseline_hull(remaining, false),
         "normal delete hull");

  tree = make_tree(dataset.points);
  expect(tree->insert(dataset.pivot_insert), "pivot insert accepted");
  auto with_pivot_insert = dataset.points;
  with_pivot_insert.push_back(dataset.pivot_insert);
  expect(tree->valid(), "pivot insert invariants");
  expect(tree->hull(false) == baseline_hull(with_pivot_insert, false),
         "pivot insert hull");

  tree = make_tree(dataset.points);
  expect(tree->erase(dataset.pivot), "pivot delete accepted");
  auto without_pivot = dataset.points;
  without_pivot.erase(
      std::remove(without_pivot.begin(), without_pivot.end(), dataset.pivot),
      without_pivot.end());
  expect(tree->valid(), "pivot delete invariants");
  expect(tree->hull(false) == baseline_hull(without_pivot, false),
         "pivot delete hull");
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

void test_benchmark_smoke() {
  const BenchmarkConfig config{32, 0xC0FFEE, 0, 2, 4};
  const auto dataset = generate_dataset(config);
  expect(dataset.points.size() == 32, "benchmark dataset size");
  expect(std::set<std::pair<long long, long long>>(
             dataset.coordinates.begin(), dataset.coordinates.end())
             .size() == 32,
         "benchmark dataset uniqueness");
  verify_workloads(dataset);
  const auto reports = run_benchmarks(dataset, config);
  expect(reports.size() == 7, "benchmark workload count");
  for (const auto &report : reports)
    expect(report.samples.size() == 2, "benchmark sample count");
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
    test_benchmark_smoke();
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

  }

  if (run_benchmark) {
    const BenchmarkConfig config{100, 0xC0FFEE, 1, 31, 50};
    const auto dataset = generate_dataset(config);
    verify_workloads(dataset);
    (void)run_benchmarks(dataset, config);
  }

  return 0;
}
