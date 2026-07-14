# Dynamic Convex Hull with AVL Tree

A C++17 implementation of a dynamic point set maintaining convex hulls using an AVL tree sorted in Graham-scan order. Points are stored relative to a pivot with exact integer arithmetic (`__int128`) to avoid floating-point rounding errors.

## Build

### Make

```bash
make
```

### CMake

```bash
cmake -S . -B build/cmake
cmake --build build/cmake
```

## Run

### Self-Test

Validates insertion, deletion, pivot changes, and hull construction:

```bash
build/make/hull --self-test
```

Or run via CMake:

```bash
ctest --test-dir build/cmake --output-on-failure
```

### Benchmark

Benchmarks 7 workload types across batch Graham scan, AVL build + hull, hull query, normal insert/delete, and pivot-changing insert/delete. After three warm-ups, each workload records 101 runs and reports total time, mean with sample standard deviation, min, median, max, p75, p95, and p99:

```bash
build/make/hull --benchmark
```

## Architecture

- **Pivot Management**: First inserted point becomes pivot; all others stored relative to pivot as (dx, dy) offset
- **Ordering**: Graham-scan polar order with 3-tier comparator: (1) upper-half plane test, (2) cross product sign, (3) squared distance
- **Balancing**: AVL tree with height tracking, left/right rotations, and balance factor maintenance
- **Exact Geometry**: `__int128` cross products and squared distances prevent floating-point rounding
- **Lazy Rebuild**: When new point would become pivot (y,x) < current pivot (y,x), all n points are re-inserted O(n log n)

## Files

- `include/avl.hpp`: Public interface (Point, DynamicHull class)
- `src/avl.cpp`: AVL tree implementation (rotations, rebalancing, insertion, deletion)
- `src/main.cpp`: Self-tests, baseline Graham scan, benchmark harness
- `Makefile`: Make build producing `build/make/hull`
- `CMakeLists.txt`: CMake build and CTest registration producing `build/cmake/hull`

## Performance Characteristics

- **Insert**: O(log n) normal, O(n log n) when pivot changes
- **Delete**: O(log n) normal, O(n log n) when deleting pivot
- **Hull**: O(n) after points are in sorted order
- **Space**: O(n) for n points plus O(1) overhead per node

## Implementation Notes

- Node stores: Point, left/right pointers, height
- Both baseline and AVL implementations use identical Graham-scan stack algorithm
- Exact arithmetic ensures deterministic results independent of platform or compiler

