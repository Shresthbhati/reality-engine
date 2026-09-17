"""Temporary test runner that writes results to a file."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_room_inference.py", "tests/test_world_compiler.py", "tests/test_geometric_reasoning.py", "tests/test_reconstruction_orchestrator.py", "tests/test_compile_command.py", "-q", "--tb=line"],
    capture_output=True, text=True
)

with open("test_results.txt", "w") as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\nSTDERR:\n")
    f.write(result.stderr)
    f.write(f"\nReturn code: {result.returncode}\n")

print("Done. Results written to test_results.txt")
