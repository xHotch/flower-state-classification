from subprocess import Popen
import sys

"""
This script is used to run the flower state classification pipeline continuously upon crashing.
"""

def main():
    args = sys.argv[1:]
    while True:
        print("Starting Flower State Pipeline")
        p = Popen(["python", 'src/flower_state_classification/run.py '] + args)
        p.wait()


if __name__ == "__main__":
    main()