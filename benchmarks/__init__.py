"""Reality Engine benchmarks.

A regular package on purpose: ``dash`` (a dependency of Open3D) ships its own top-level ``benchmarks`` package, and a bare
namespace folder loses to it, which broke ``import benchmarks.<module>`` for every benchmark test once Open3D was installed.
"""
