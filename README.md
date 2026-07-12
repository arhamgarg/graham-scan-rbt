# Dynamic Convex Hull with Red-Black Tree

A C++17 implementation of a dynamic point set maintaining convex hulls using a Red-Black Tree sorted in Graham-scan order. Points are stored relative to a pivot with exact integer arithmetic (__int128) to avoid floating-point rounding errors.

## Build

```bash
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic src/main.cpp src/rbt.cpp -o hull
```

## Run

### Self-Test
Validates insertion, deletion, pivot changes, hull construction, and allocation tracking:
```bash
./hull --self-test
```

### Benchmark
Runs reproducible timing benchmarks across 5 scenarios (baseline sort, normal insert, normal delete, pivot-changing insert, pivot deletion) with 31 samples each:
```bash
./hull --benchmark
```

## Profiling

### Linux

System-wide performance counters:
```bash
perf stat ./hull --benchmark
```

Detailed call graph recording:
```bash
perf record -g ./hull --benchmark
perf report
```

Memory usage profiling:
```bash
valgrind --tool=massif ./hull --benchmark
ms_print massif.out.* | head -100
```

### macOS

Time profiler (CPU cycles):
```bash
xcrun xctrace record --template 'Time Profiler' --launch -- ./hull --benchmark
```

Memory allocation profiler:
```bash
xcrun xctrace record --template Allocations --launch -- ./hull --benchmark
```

## Architecture

- **Pivot Management**: First inserted point becomes pivot; all others stored relative to pivot as (dx, dy) offset from pivot
- **Ordering**: Graham-scan polar order with 3-tier comparator: (1) upper-half plane test, (2) cross product sign, (3) squared distance
- **Invariants**: Black-root, red-parent rule, equal black-height across all paths
- **Exact Geometry**: `__int128` cross products and squared distances prevent floating-point rounding
- **Lazy Rebuild**: When new point would become pivot (y,x) < current pivot (y,x), all n points are re-inserted O(n log n)

## Files

- `src/rbt.hpp`: Public interface (Point, DynamicHull class, Color enum)
- `src/rbt.cpp`: RBT implementation (rotations, rebalancing, insertion, deletion, validation)
- `src/main.cpp`: Self-tests, baseline Graham scan, benchmark harness, allocation tracking

## Performance Characteristics

- **Insert**: O(log n) normal, O(n log n) when pivot changes
- **Delete**: O(log n) normal, O(n log n) when deleting pivot
- **Hull**: O(n) after points are in sorted order
- **Space**: O(n) for n points plus O(1) overhead per node

## Implementation Notes

- Sentinel nil_ node simplifies boundary handling in rotations
- Node stores: Point, dx, dy, squared distance (__int128), color, parent/left/right pointers
- Both baseline and RBT implementations use identical Graham-scan stack algorithm
- Canonicalization (rotation to minimum (y,x) point) enables deterministic hull comparison
