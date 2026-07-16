# Dynamic Convex Hull with AVL Trees, Red-Black Trees, and Arrays

A C++17 implementation of a dynamic point set maintaining convex hulls using three backend data structures sorted in Graham-scan order:
1. **AVL Tree**
2. **Red-Black Tree**
3. **Sorted Array/Vector**

Points are stored relative to a pivot with exact integer arithmetic (`__int128`) to avoid floating-point rounding errors. All three data structures are integrated side-by-side to allow comparative performance profiling, correctness checks, and memory allocation tracking.

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

Validates insertion, deletion, pivot changes, and hull construction invariants for RBT, AVL, and Array backends:

```bash
build/make/hull --self-test
```

Or run via CMake:

```bash
ctest --test-dir build/cmake --output-on-failure
```

### Benchmark

Benchmarks 19 workload types across batch Graham scan, RBT/AVL/Array build + hull, RBT/AVL/Array hull queries, normal insert/delete, and pivot-changing insert/delete. Reports total time, mean, standard deviation, min, median, max, percentiles, and memory metrics:

```bash
build/make/hull --benchmark
```

> [!WARNING]
> Due to the $O(N)$ insertion complexity of the sorted `std::vector` implementation, the Array/Vector benchmarks take $O(N^2)$ time to build the set. Running the benchmark with the default dataset size of 100,000 points will take several hours to complete.

## Architecture

- **Pivot Management**: First inserted point becomes pivot; all others stored relative to pivot as (dx, dy) offset.
- **Ordering**: Graham-scan polar order with 3-tier comparator: (1) upper-half plane test, (2) cross product sign, (3) squared distance.
- **Balancing Schemes / Backends**:
  - **AVL Tree**: Height-balanced tree using single/double rotations.
  - **Red-Black Tree**: Balanced using node coloring (RED/BLACK) and color fixup rotations.
  - **Array/Vector**: Stored as a sorted contiguous `std::vector` using binary search (`std::lower_bound`) for lookups.
- **Exact Geometry**: `__int128` cross products and squared distances prevent floating-point rounding errors.
- **Lazy Rebuild**: When a new point becomes the pivot (y,x) < current pivot (y,x), all $N$ points are re-inserted in $O(N \log N)$ (trees) or $O(N^2)$ (array) time.

## Files

- `include/avl.hpp` / `src/avl.cpp`: AVL tree implementation (rotations, rebalancing, insertion, deletion)
- `include/rbt.hpp` / `src/rbt.cpp`: Red-Black tree implementation (rotations, color fixups, insertion, deletion)
- `include/arr.hpp` / `src/arr.cpp`: Sorted vector/array implementation (binary search inserts and erases)
- `src/main.cpp`: Self-tests, baseline Graham scan, benchmark harness running RBT, AVL, and Array side-by-side
- `Makefile`: Make build producing `build/make/hull`
- `CMakeLists.txt`: CMake build and CTest registration producing `build/cmake/hull`

## Performance Characteristics

| Operation | AVL Tree | Red-Black Tree | Sorted Array/Vector |
| :--- | :--- | :--- | :--- |
| **Insert (Normal)** | $O(\log N)$ | $O(\log N)$ | $O(N)$ |
| **Delete (Normal)** | $O(\log N)$ | $O(\log N)$ | $O(N)$ |
| **Insert (Pivot Change)** | $O(N \log N)$ | $O(N \log N)$ | $O(N^2)$ |
| **Delete (Pivot Change)** | $O(N \log N)$ | $O(N \log N)$ | $O(N^2)$ |
| **Hull Query** | $O(N)$ | $O(N)$ | $O(N)$ |
| **Space Complexity** | $O(N)$ | $O(N)$ | $O(N)$ (no pointer overhead) |
