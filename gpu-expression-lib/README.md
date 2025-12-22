# Spark GPU Expression Library

A standalone library for GPU-accelerating Apache Spark SQL expressions.

> **Note**: This library lives within the `spark-rapids-jni` repository on the `gpu-expression-lib` branch,
> but builds and operates independently from the main spark-rapids-jni code.

## Overview

This project provides two completely independent components:

### Java/Scala Component (`java/`)

A pure Java library that determines if a Spark SQL expression can be translated to run on GPU.

- **No native dependencies** - pure Java/Scala code
- Depends only on Spark SQL (provided scope)
- **Per-expression APIs** for maximum flexibility in intermixing with other GPU implementations
- Compatible with external projects like Gluten

### Native C++ Component (`native/`)

A standalone CUDA/cudf library for GPU expression execution.

- **No Java/JNI dependencies** - pure C++ with CUDA
- **Global init/shutdown APIs** for proper memory management
- **Custom memory allocator support** to integrate with your memory manager
- **Logical type system** independent of Spark JVM types
- Can be called from any language supporting C ABI

## Supported Expressions

| Expression | Java Support Check | C++ GPU Execution |
|------------|-------------------|-------------------|
| `Round` | ✅ | ✅ |
| `BRound` | ✅ | ✅ |
| `DateFormatClass` | ✅ | ✅ |

## Requirements

### Target Spark Version

This library targets **Apache Spark 4.0.x**.

### Java Component Requirements

| Requirement | Minimum Version |
|-------------|-----------------|
| Java | 17+ |
| Maven | 3.6+ |
| Spark | 4.0.0 (provided at runtime) |

### Native Component Requirements

**For Docker builds (recommended):**

| Requirement | Version |
|-------------|---------|
| Docker | 20.10+ |
| NVIDIA Container Toolkit | Latest |
| NVIDIA GPU | Compute Capability 7.0+ (Volta or newer) |

**For local builds:**

| Requirement | Minimum Version |
|-------------|-----------------|
| CUDA Toolkit | 12.0+ |
| CMake | 3.26.4+ |
| GCC | 11+ (gcc-toolset-14 recommended) |
| Ninja | 1.10+ (optional, improves build speed) |

## Initial Setup

### Clone and Switch to Branch

The gpu-expression-lib is on a dedicated branch in the spark-rapids-jni repository:

```bash
# Clone spark-rapids-jni with submodules
git clone --recurse-submodules https://github.com/NVIDIA/spark-rapids-jni.git
cd spark-rapids-jni

# Switch to the gpu-expression-lib branch
git checkout gpu-expression-lib

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

This will populate `thirdparty/cudf` with the cudf source code (shared with spark-rapids-jni).

## Building

### Java Component

```bash
# From the gpu-expression-lib directory
cd gpu-expression-lib

# Set JAVA_HOME to Java 17+
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Build and test
mvn clean package

# Run tests only
mvn test
```

### Native Component (Docker - Recommended)

The Docker build provides a reproducible environment with CUDA and build tools.

**Build cudf and the library inside Docker:**

```bash
# From the gpu-expression-lib directory
cd gpu-expression-lib
./build/run-in-docker ./build/buildcpp.sh
```

This will automatically build the Docker image if it doesn't exist, then run the build.

This will:
1. Build libcudf from source (static library)
2. Build libspark_gpu_expr against cudf
3. Run tests (if enabled)

**Interactive shell (for debugging):**

```bash
./build/run-in-docker
# Then inside the container:
./build/buildcpp.sh
```

### Native Component (Local Build)

Requires CUDA toolkit to be installed locally. cudf will be built from source.

```bash
./build/buildcpp.sh
```

### Native Build Configuration

The build script (`build/buildcpp.sh`) supports these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDF_BUILD_TYPE` | `Release` | Build type: `Release`, `Debug`, `RelWithDebInfo` |
| `CMAKE_CUDA_ARCHITECTURES` | `RAPIDS` | CUDA architectures to build for (see below) |
| `BUILD_TESTS` | `ON` | Build and run unit tests: `ON` or `OFF` |
| `CPP_PARALLEL_LEVEL` | `10` | Number of parallel compile jobs |
| `CMAKE_GENERATOR` | `Ninja` | CMake generator: `Ninja` or `Unix Makefiles` |
| `CUDF_PATH` | `thirdparty/cudf` | Path to cudf source |
| `LIBCUDF_DEPENDENCY_MODE` | `pinned` | Use pinned (`pinned`) or latest (`latest`) dependency versions |

**Examples:**

```bash
# Debug build without tests
CUDF_BUILD_TYPE=Debug BUILD_TESTS=OFF ./build/buildcpp.sh

# Build for specific GPU architectures (for distribution)
CMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" ./build/buildcpp.sh

# Build for the GPU in the current machine only
CMAKE_CUDA_ARCHITECTURES=native ./build/buildcpp.sh

# Skip cudf reconfiguration (faster rebuilds)
LIBCUDF_BUILD_CONFIGURE=false ./build/buildcpp.sh
```

**CUDA Architecture Values:**

| Value | Description |
|-------|-------------|
| `RAPIDS` | Use rapids-cmake to determine architectures (default) |
| `native` | Auto-detect GPU in build machine (fast build, not portable) |
| `all` | All supported architectures (slow build, maximum compatibility) |
| `all-major` | All major architectures |
| `70` | Volta (V100) |
| `75` | Turing (T4, RTX 20xx) |
| `80` | Ampere (A100, RTX 30xx) |
| `86` | Ampere (RTX 30xx mobile, A10) |
| `89` | Ada Lovelace (RTX 40xx, L4) |
| `90` | Hopper (H100) |
| `70;80;90` | Semicolon-separated list for multiple architectures |

**Output:**

The built library will be at `cpp-target/native/cmake-build/libspark_gpu_expr.so`.

Note: The C++ build uses `cpp-target/` while Maven uses `target/`. This allows independent management of the two build systems.

## Usage

### Java API: Per-Expression Support Checking

Use individual APIs for maximum flexibility in intermixing with other GPU implementations:

```java
import com.nvidia.spark.expression.RoundSupport;
import com.nvidia.spark.expression.DateFormatSupport;

// Check specific expression types
if (expr instanceof Round || expr instanceof BRound) {
    boolean canRun = RoundSupport.canTranslate(expr);
}

if (expr instanceof DateFormatClass) {
    boolean canRun = DateFormatSupport.canTranslate(expr);
}
```

Or use the convenience aggregated API:

```java
import com.nvidia.spark.expression.ExpressionSupport;

// Check any supported expression
boolean canRun = ExpressionSupport.canTranslate(myExpression);
```

### C++ API: Initialization

```cpp
#include <spark_gpu_expr/init.hpp>

// Initialize with defaults
spark_gpu_expr::init();

// Or with custom options
spark_gpu_expr::InitOptions options;
options.device_id = 0;
options.enable_memory_pool = true;
options.initial_pool_size = 1024 * 1024 * 1024;  // 1GB
spark_gpu_expr::init(options);

// ... use library ...

// Clean shutdown - releases cached resources
spark_gpu_expr::shutdown();
```

### C++ API: Custom Memory Allocation

```cpp
#include <spark_gpu_expr/memory.hpp>

class MyMemoryResource : public spark_gpu_expr::MemoryResource {
public:
    void* allocate(std::size_t bytes, std::size_t alignment) override {
        // Your allocation logic
    }
    void deallocate(void* ptr, std::size_t bytes, std::size_t alignment) override {
        // Your deallocation logic
    }
};

MyMemoryResource my_mr;
spark_gpu_expr::InitOptions opts;
opts.memory_resource = &my_mr;
spark_gpu_expr::init(opts);
```

### C++ API: Logical Types

```cpp
#include <spark_gpu_expr/logical_type.hpp>

// Create types independent of Spark JVM
auto decimal = spark_gpu_expr::LogicalType::decimal_type(10, 2);
auto array = spark_gpu_expr::LogicalType::array_type(
    spark_gpu_expr::LogicalType::string_type()
);

// Query properties
bool numeric = decimal->is_numeric();  // true
std::string name = decimal->to_string();  // "decimal(10,2)"
```

### C++ API: Expression Execution

```cpp
#include <spark_gpu_expr/round.hpp>
#include <spark_gpu_expr/date_format.hpp>

// Round values
spark_gpu_expr::RoundParams params{
    .target_scale = 2,
    .mode = spark_gpu_expr::RoundMode::HALF_UP,
    .output_type = spark_gpu_expr::OutputType::float64(),
    .overflow_policy = spark_gpu_expr::OverflowPolicy::NULLIFY
};
auto result = spark_gpu_expr::round(input_col, params);

// Format timestamps
spark_gpu_expr::DateFormatParams fmt_params{
    .format = "yyyy-MM-dd HH:mm:ss",
    .timezone = "UTC"
};
auto formatted = spark_gpu_expr::date_format(timestamp_col, fmt_params);
```

## Testing

### Java Tests

```bash
mvn test
```

### C++ Tests

```bash
# Tests run automatically during build if BUILD_TESTS=ON (default)
./build/buildcpp.sh

# Skip tests
BUILD_TESTS=OFF ./build/buildcpp.sh
```

## Project Structure

```
spark-rapids-jni/                  # Parent repository (gpu-expression-lib branch)
├── gpu-expression-lib/            # This library
│   ├── pom.xml                    # Maven build for Java component
│   ├── java/                      # Java/Scala component
│   │   └── src/
│   │       ├── main/java/         # Main source code
│   │       └── test/java/         # Unit tests
│   ├── native/                    # C++ component
│   │   ├── CMakeLists.txt         # CMake build (uses rapids-cmake)
│   │   ├── include/               # Public headers
│   │   │   └── spark_gpu_expr/
│   │   │       ├── init.hpp       # Global init/shutdown
│   │   │       ├── memory.hpp     # Memory resource interfaces
│   │   │       ├── logical_type.hpp  # Logical type system
│   │   │       ├── types.hpp      # OutputType, RoundMode, etc.
│   │   │       ├── round.hpp      # Round expression
│   │   │       └── date_format.hpp   # DateFormat expression
│   │   ├── src/                   # Source files
│   │   └── tests/                 # Unit tests
│   ├── target/                    # Maven build output (Java)
│   ├── cpp-target/                # C++ build output (native)
│   ├── build/                     # Build scripts
│   │   ├── run-in-docker          # Run commands in Docker (builds image if needed)
│   │   └── buildcpp.sh            # Native build script (builds cudf + this library)
│   └── ci/
│       └── Dockerfile             # Build environment
├── thirdparty/
│   ├── cudf/                      # Shared cudf submodule (GPU DataFrame library)
│   └── cudf-pins/                 # Dependency version pinning
├── src/                           # Original spark-rapids-jni code (untouched)
│   └── ...
└── pom.xml                        # Original spark-rapids-jni pom (untouched)
```

## License

Apache License 2.0 - See LICENSE file.

## Key Design Decisions

### No JNI

The Java and C++ components are completely independent:
- Java code cannot call native code
- C++ code cannot call Java code
- Each can be built, tested, and deployed independently

This design enables:
- Clean separation of concerns
- Independent testing
- Flexible deployment options
- Integration with systems that have their own native execution engines

### Per-Expression APIs

The Java component provides separate static APIs per expression type, allowing:
- Maximum flexibility to intermix CUDF expressions with other GPU implementations (like TQP)
- No central registry that must be kept in sync
- Easy extension for new expressions

### GPU Expression Execution

The C++ component provides GPU kernels from cudf:
- Custom floating-point rounding matching Spark semantics
- Timestamp to string formatting via cudf
- Full control over memory allocation
- Logical type system independent of JVM types

### Build System

The native build system mirrors spark-rapids-jni:
- cudf is included as a git submodule
- rapids-cmake handles dependency management
- Pinned dependency versions for reproducible builds
- Docker container for consistent build environment
