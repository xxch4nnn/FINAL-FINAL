"""
labeler.py
Implements the State Machine logic to transform binary labels into 4-state labels.
"""

import pandas as pd
import numpy as np

class StateLabeler:
    """
    Transforms binary 'is_pressed' sequence into 4-state labels:
    0: Hover
    1: Press (Transition 0->1)
    2: Hold (1->1)
    3: Release (1->0)
    """

    HOVER = 0
    PRESS = 1
    HOLD = 2
    RELEASE = 3

    @staticmethod
    def apply(df: pd.DataFrame, binary_col: str = 'is_pressed') -> pd.DataFrame:
        """
        Adds 'state_label' column to the DataFrame.
        Assumes DataFrame is sorted by time and represents a SINGLE sequence.
        """
        if df.empty:
            return df

        # Ensure we are working on a copy to avoid SettingWithCopy warnings
        df = df.copy()

        curr = df[binary_col]
        prev = df[binary_col].shift(1).fillna(0) # Assume start with Hover (0)

        # State Logic
        # 0 -> 0 : Hover
        # 0 -> 1 : Press
        # 1 -> 1 : Hold
        # 1 -> 0 : Release

        conditions = [
            (prev == 0) & (curr == 0),
            (prev == 0) & (curr == 1),
            (prev == 1) & (curr == 1),
            (prev == 1) & (curr == 0)
        ]

        choices = [
            StateLabeler.HOVER,
            StateLabeler.PRESS,
            StateLabeler.HOLD,
            StateLabeler.RELEASE
        ]

        df['state_label'] = np.select(conditions, choices, default=StateLabeler.HOVER)

        # Cleanup: 'Release' state (1->0) actually occurs on the frame where current is 0.
        # This is logically correct (the frame it becomes released).
        # 'Press' state (0->1) occurs on the frame where current is 1.

        return df
