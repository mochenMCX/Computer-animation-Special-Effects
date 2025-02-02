#pragma once
// https://stackoverflow.com/a/53365539/10187092
// Check for feature test macro for <filesystem>
#if defined(__cpp_lib_filesystem)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0

// Check for feature test macro for <experimental/filesystem>
#elif defined(__cpp_lib_experimental_filesystem)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// We can't check if headers exist...
// Let's assume experimental to be safe
#elif !defined(__has_include)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// Check if the header "<filesystem>" exists
#elif __has_include(<filesystem>)

// If we're compiling on Visual Studio and are not compiling with C++17, we need
// to use experimental
#ifdef _MSC_VER

// Check and include header that defines "_HAS_CXX17"
#if __has_include(<yvals_core.h>)
#include <yvals_core.h>

// Check for enabled C++17 support
#if defined(_HAS_CXX17) && _HAS_CXX17
// We're using C++17, so let's use the normal version
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0
#endif
#endif

// If the marco isn't defined yet, that means any of the other VS specific
// checks failed, so we need to use experimental
#ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1
#endif

// Not on Visual Studio. Let's use the normal version
#else  // #ifdef _MSC_VER
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0
#endif

// Check if the header "<filesystem>" exists
#elif __has_include(<experimental/filesystem>)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// Fail if neither header is available with a nice error message
#else
#error Could not find system header "<filesystem>" or "<experimental/filesystem>"
#endif

// We priously determined that we need the exprimental version
#if INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
#include <experimental/filesystem>
namespace util {
namespace fs = std::experimental::filesystem;
}
// We have a decent compiler and can use the normal version
#else
#include <filesystem>
namespace util {
namespace fs = std::filesystem;
}
#endif  // INCLUDE_STD_FILESYSTEM_EXPERIMENTAL

namespace util {
class PathFinder final {
 public:
    // delete all constructors
    PathFinder() = delete;
    PathFinder(const PathFinder&) = delete;
    PathFinder(PathFinder&&) = delete;
    // delete destructors
    ~PathFinder() = delete;
    // delete assignment operator
    PathFinder& operator=(const PathFinder&) = delete;
    PathFinder& operator=(PathFinder&&) = delete;
    // Find asset folder by specific filename
    static bool initialize();
    // Make relative path to asset folder to absolute path
    static fs::path find(const char* filename);

 private:
    static fs::path assetPath;
};
}  // namespace util
