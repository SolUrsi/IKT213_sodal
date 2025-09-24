import subprocess
import sys

# Run all the .py files

def run(script):
    process = subprocess.Popen([sys.executable, script])
    process.wait()

run("test_orb_bf.py")
run("test_sift_flann.py")
run("real_orb_bf.py")
run("real_sift_flann.py")
