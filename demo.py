"""End-to-end synthetic-only checkpoint. Never discovers or scans real drives."""
import config
from generate_data import generate
from train import train
from inference import scan, print_inspections, Predictor


def main():
    generate()
    train()
    predictor = Predictor()
    print("\nMODE 1: synthetic controlled scan")
    print_inspections(scan([config.TEST_DIRECTORY], predictor=predictor), approved=True)
    print("\nMODE 2: synthetic simulation of full metadata pipeline (no drive discovery)")
    candidates = scan([config.TEST_DIRECTORY], full=True, predictor=predictor)
    print("Demo explicitly approves synthetic content inspection.")
    print_inspections(candidates, approved=True)


if __name__ == "__main__":
    main()
