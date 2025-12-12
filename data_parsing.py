import pandas as pd
import numpy as np

class DataParsing: 

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.result:pd.DataFrame 
        self.result['id'] = self.df['id']
        self.result['size'] = self.df['size']

    def get_csp(self) -> pd.DataFrame:
        return self.result
    
    def next(self): 
        pass

    def _variables(self):
        pass
    def _domains(self):
        pass
    def _constraints(self):
        pass

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
    """Constraint for 'x is one House away from y'"""
    def __init__(self, x, y, distance:int=0):
        super().__init__()

    def is_satisfied(self):
        return super().is_satisfied()
    
class BetweenContraint(Constraint):
    """Constraint for 'between x and y is one House'"""
    def __init__(self):
        super().__init__()

    def is_satisfied(self):
        return super().is_satisfied()

class RightConstraint(Constraint):
    """Constraint for 'x is right of y'"""
    def __init__(self, x, y, distance:int=0):
        super().__init__()

    def is_satisfied(self):
        return super().is_satisfied()
    
if __name__ == "__main__":
    dp = DataParsing(pd.read_parquet("data/Gridmode-00000-of-00001.parquet"))
    print(dp.get_csp())