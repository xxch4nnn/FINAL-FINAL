import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from feature_engine import LiveFeatureExtractor, FEATURE_COLUMNS, construct_feature_row
import numpy as np

def test_congruency():
    print("Running Congruency Audit...")

    # 1. Instantiate Live Extractor
    extractor = LiveFeatureExtractor()
    print(f"LiveFeatureExtractor initialized.")
    print(f"Expected Feature Count: {len(FEATURE_COLUMNS)}")
    print(f"Live Extractor Feature Names: {len(extractor.feature_names)}")

    # Verify names match exactly
    if extractor.feature_names != FEATURE_COLUMNS:
        print("FAIL: LiveFeatureExtractor columns do not match global FEATURE_COLUMNS.")
        diff = set(extractor.feature_names) ^ set(FEATURE_COLUMNS)
        print(f"Diff: {diff}")
        sys.exit(1)
    else:
        print("PASS: Column Definitions Match.")

    # 2. Simulate Data Flow
    # Fill buffer
    print("Simulating Data Flow...")
    class MockLandmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    for i in range(10):
        lm = [MockLandmark(0.5, 0.5, 0.0) for _ in range(21)]
        # Add some motion
        lm[8].x += i * 0.01
        lm[8].z -= i * 0.001 # moving deeper
        extractor.update(lm)

    # Extract
    vec = extractor.extract()

    if vec is None:
        print("FAIL: No vector extracted after 10 frames.")
        sys.exit(1)

    if vec.shape[1] != len(FEATURE_COLUMNS):
        print(f"FAIL: Extracted vector shape {vec.shape} mismatches expected columns {len(FEATURE_COLUMNS)}.")
        sys.exit(1)

    print(f"PASS: Vector extraction successful. Shape: {vec.shape}")
    print("Audit Complete: System is Congruent.")

if __name__ == "__main__":
    test_congruency()
