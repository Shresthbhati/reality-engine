import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_room_inference.py", "-q", "--tb=line"], capture_output=True, text=True)
with open("quick_test_out.txt", "w") as f:
    f.write(r.stdout + "\n" + r.stderr)
print("exit:", r.returncode)
