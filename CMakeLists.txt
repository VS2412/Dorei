cmake_minimum_required(VERSION 3.20)
project(ai_agent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FVAD REQUIRED libfvad)
pkg_check_modules(ONNXRUNTIME REQUIRED libonnxruntime)

set(WHISPER_DIR "/home/Aurelius/Documents/AdoVs/whisper.cpp"
    CACHE PATH "Path to whisper.cpp source tree (needs include/ and build/ subdirs)")

# All of ARIA except main() — shared by the daemon binary and the bench
# harness so benchmarks exercise the exact production code paths.
add_library(aria_core STATIC
    src/daemon.cpp
    src/logger.cpp
    src/recorder.cpp
    src/transcriber.cpp
    src/llm.cpp
    src/executor.cpp
    src/tts.cpp
    src/context.cpp
    src/memory.cpp
    src/intent.cpp
    src/screen.cpp
    src/timer.cpp
    src/config.cpp
    src/wakeword.cpp
    src/vision.cpp
    src/planner.cpp
)

target_include_directories(aria_core PUBLIC
    src
    ${WHISPER_DIR}/include
    ${WHISPER_DIR}/ggml/include
    ${FVAD_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_directories(aria_core PUBLIC
    ${WHISPER_DIR}/build/src
    ${WHISPER_DIR}/build/ggml/src
    ${ONNXRUNTIME_LIBRARY_DIRS}
)

target_link_libraries(aria_core PUBLIC
    whisper ggml
    CURL::libcurl
    nlohmann_json::nlohmann_json
    SQLite3::SQLite3
    ${FVAD_LIBRARIES}
    ${ONNXRUNTIME_LIBRARIES}
)

target_compile_options(aria_core PRIVATE -Wall -Wextra -O2 ${FVAD_CFLAGS_OTHER})
target_compile_definitions(aria_core PUBLIC _POSIX_C_SOURCE=200809L)

add_executable(ai-agent src/main.cpp)
target_link_libraries(ai-agent PRIVATE aria_core)
target_compile_options(ai-agent PRIVATE -Wall -Wextra -O2)

# Phase 0 bench harness — `aria-bench core` (text → action/TTFT, no microphone)
# and `aria-bench asr` (WAV → transcript accuracy/latency).
add_executable(aria-bench bench/bench.cpp)
target_link_libraries(aria-bench PRIVATE aria_core)
target_compile_options(aria-bench PRIVATE -Wall -Wextra -O2)

enable_testing()
add_subdirectory(arise)
