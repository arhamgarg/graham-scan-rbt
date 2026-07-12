#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

#include "rbt.hpp"

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

int main(int argc, char *argv[]) {
  assert(cross({0, 0}, {1, 0}, {0, 1}) > 0);
  assert((Point{2, 3} == Point{2, 3}));

  bool run_self_test = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--self-test") == 0) {
      run_self_test = true;
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
  }

  return 0;
}
