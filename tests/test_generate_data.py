import importlib
import os
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

from src.generate_data import simulate_battery, N_CYCLES, N_BATTERIES, EOL_THRESHOLD, main


class TestGenerateData(unittest.TestCase):

    def test_reproducibility_same_seed(self):
        """Verify that identical seeds produce identical simulated data and parameters."""
        df1, params1 = simulate_battery(1, seed=42)
        df2, params2 = simulate_battery(1, seed=42)

        pd.testing.assert_frame_equal(df1, df2)
        self.assertEqual(params1, params2)

    def test_variability_different_seeds(self):
        """Verify that different seeds produce different simulated measurements."""
        df1, _ = simulate_battery(1, seed=42)
        df2, _ = simulate_battery(1, seed=99)

        # Ensure SOH measurements differ between different seeds
        self.assertFalse(
            np.allclose(df1["SOH"].values, df2["SOH"].values),
            "Expected different seeds to produce different SOH trajectories",
        )

    def test_schema_and_columns(self):
        """Verify that the generated DataFrame contains the expected columns and formatting."""
        expected_columns = ["battery_id", "cycle", "SOH", "capacity_Ah"]
        df, params = simulate_battery(3, seed=10)

        self.assertEqual(list(df.columns), expected_columns)
        self.assertEqual(df["battery_id"].iloc[0], "B0003")
        self.assertEqual(len(df), N_CYCLES)
        self.assertEqual(df["cycle"].iloc[0], 1)
        self.assertEqual(df["cycle"].iloc[-1], N_CYCLES)

        # Check parameter dictionary keys
        expected_param_keys = {"a", "b", "c", "d", "noise_std"}
        self.assertEqual(set(params.keys()), expected_param_keys)

    def test_soh_bounds_and_capacity(self):
        """Verify that SOH is clipped to valid physical bounds and capacity scales nominally."""
        df, _ = simulate_battery(2, seed=77)

        self.assertTrue((df["SOH"] >= 0.5).all(), "SOH dropped below physical minimum bound of 0.5")
        self.assertTrue((df["SOH"] <= 1.0).all(), "SOH exceeded physical maximum bound of 1.0")

        # Capacity should match 2.0 * SOH
        np.testing.assert_allclose(df["capacity_Ah"].values, df["SOH"].values * 2.0)

    def test_import_has_no_filesystem_side_effects(self):
        """Verify that importing src.generate_data does not write to the filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_csv = os.path.join(tmpdir, "test.csv")
            # Reloading module should not trigger writing to data/
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                import src.generate_data
                importlib.reload(src.generate_data)
                mock_to_csv.assert_not_called()

    def test_main_function(self):
        """Verify that main() generates the full multi-battery dataset with expected row count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_csv = os.path.join(tmpdir, "battery_degradation.csv")
            with patch("os.path.join", return_value=fake_csv), patch("os.makedirs"), patch("builtins.print"):
                dataset, true_params = main()

                self.assertEqual(len(dataset), N_BATTERIES * N_CYCLES)
                self.assertEqual(len(true_params), N_BATTERIES)
                self.assertEqual(dataset["battery_id"].nunique(), N_BATTERIES)


if __name__ == "__main__":
    unittest.main()
