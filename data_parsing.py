import pandas as pd
import numpy as np
import re

class DataParsing: 

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.result = pd.DataFrame() 
        self.result['id'] = self.df['id']
        self.result['size'] = self.df['size']

    def get_csp(self) -> pd.DataFrame:
        return self.result

    def _variables(self):
        pass
    def _domains(self):
        pass
    def _constraints(self):     
        word_num = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'first':1, 'second':2, 'third':3, 'fourth':4, 'fifth':5, 'sixt':6, 'seventh':7, 'ninth':9, 'tenth':10}
        puzzles = self.df.puzzle 
        clue_list = []
        constraint_list = []

        
        for puzzle in puzzles:
            
            #filter all constraints using '1.' etc.
            for i in puzzle.splitlines(): 
                if re.match(r'^[0-9]+[.]', i):
                    clue_list.append(i)

            #sort constraints
            for con in clue_list: 
                #TODO rework section below to a match case

                match(con):
                    #leftConstraint
                    #- clue contains 'left'
                    #- clue can be 'somewhere to the left' or 'directly left of' 
                    case _ if ' left' in con:
                        if ' directly ' in con:
                            #direct left constraint
                            print(f'directly left: {con}')
                        elif ' somewhere ' in con: 
                            #somewhere left constraint
                            print(f'somewhere left: {con}')
                    
                    #rightConstraint same as left constraint
                    case _ if ' right' in con: 
                        if ' somewhere ' in con: 
                            #somewhere right constraint
                            print(f'somewhere right: {con}')

                        #there shuoldn't be a 'directly right' constraint. Leaving it just in case.    
                        elif ' directly ' in con:
                            #direct right constraint
                            print(f'directly right: {con}')
                    
                    #nextToConstraint: 
                    case _ if ' next to each other' in con: 
                        print(f'next to each other: {con}')

                    #betweenConstraint
                    case _ if ' between' in con: 
                        print(f'between: {con}')
                
                    #isNotConstraint
                    case _ if ' is not ' in con: 
                        print(f'is not: {con}')
                        
                    #isConstraint
                    case _: 
                        print(f'is: {con}')
                

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
    
class IsNotConstraint(Constraint):
    """Constraint for 'x is not y' """
    def __init__(self):
        super().__init__()
    
    def is_satisfied(self):
        return super().is_satisfied()
    
class LeftOrRightConstraint(Constraint):
    """Constraint for 'x is next to y'"""
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
    print(dp._constraints())