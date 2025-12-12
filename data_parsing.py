import pandas as pd
import numpy as np

class DataParsing: 
    def __init__(self, df: pd.DataFrame) -> None:
        pass

    def get_csp(self) -> pd.DataFrame:
        raise(NotImplementedError)


class Constraint: 
    """Parent Class for all Constraints"""
    def __init__(self):
        pass

    def is_satisfied(self):
        pass


class LeftConstraint(Constraint):
    """Constraint for 'x is left of y'"""
    def __init__(self, x, y, distance:int=0):
        pass
    
    def is_satisfied(self):
        return super().is_satisfied()
    
class IsContraint(Constraint):
    """Constraint for 'x is y' """
    def __init__(self):
        super().__init__()

    def is_satisfied(self):
        return super().is_satisfied()
    
class LeftOrRightConstraint(Constraint):
    """Constraint for 'between x and y is one House'"""
    def __init__(self, x, y, distance:int=0):
        super().__init__()

    def is_satisfied(self):
        return super().is_satisfied()

class RightConstraint(Constraint):
    """Constraint for 'x is right of y'"""
    def __init__(self, x, y, distance:int=0):
        super().__init__()

    def is_satisfied(self):
        return super().is_satisfied()