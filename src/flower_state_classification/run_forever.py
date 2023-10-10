from subprocess import Popen
import sys

PYTHON_PATH = r"C:\Users\hochi\.virtualenvs\flower-state-classification-UHBcljiH\Scripts\python.exe"

def main():
    args = sys.argv[1:]
    while True:
        print("Starting Flower State Pipeline")
        p = Popen([f"{PYTHON_PATH}", 'src/flower_state_classification/run.py '] + args)
        p.wait()


if __name__ == "__main__":
    main()