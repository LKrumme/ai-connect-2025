import pandas as pd
import numpy as np
import re

class DataParsing: 

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.result = pd.DataFrame(columns=['id', 'size', 'variables', 'domains', 'constraints']) 
        self.result['id'] = self.df['id']
        self.result['size'] = self.df['size']
        

    def get_csp(self) -> pd.DataFrame:
        self._variables()
        self._domains()
        self._constraints()
        return self.result

    def _variables(self):
        self.result['variables'] = self.df["solution"].apply(lambda x: x.get("header", None))

    def _domains(self):
        # The current logic assumes the lists in the text are always in the same order
        # as the 'variables' list. In some puzzles, these are shuffled (e.g., Colors comes before Names).
        # We should use keywords (like checking if 'Red' is in the list) to match the correct list to the correct variable.
        # Otherwise, we end up assigning 'Red' to 'Name' and the solver breaks.

        for index, problem in enumerate(self.df["puzzle"]):
            vars = self.result.loc[index, "variables"]

            # output Directionairy
            results = {}

            i = 0
            # "House" doesn't have given Values, so we generate them ourselves
            if(vars[i] == "House"):
                houses = []

                # Generates the Domain for houses
                for h_Count in range(int(self.df["size"][index][0])):
                    houses.append(h_Count+1)
                    results["House"] = houses

            for line in problem.splitlines():
                if line.strip().startswith("-"):
                    
                    i = i+1
                    # Filters all Expressions, that are in Between of '
                    # resulting List gets addedd as a new Domain
                    names = re.findall(r"`([^`]*)`", line)
                    results[vars[i]] = names
                
                    

            self.result.at[index, "domains"] = results

    def _constraints(self):     
        word_num = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'first':1, 'second':2, 'third':3, 'fourth':4, 'fifth':5, 'sixth':6, 'seventh':7, 'ninth':9, 'tenth':10}

        puzzles = self.df['puzzle']
        domains = self.result.domains

        for index, puzzle in enumerate(puzzles):
            clue_list = []
            constraint_list = []    
            curr_domain = [item for sublist in domains[index].values() for item in sublist] #flatten nested list

            curr_domain_short = [str(x).lower()[:3] for x in curr_domain]
            #filter all constraints using '1.' etc.
            for i in puzzle.splitlines(): 
                if re.match(r'^[0-9]+[.]', i):
                    clue_list.append(i[3:])

            #sort constraints
            for con in clue_list:

                first = None
                second = None
                tmp_distance=None

                #TODO Later
                # Replace the 3-character slicing logic with Length-Priority Matching
                # 1. Sort `curr_domain` by length (descending).
                # 2. Iterate through the domain and check `if item in constraint_string`.
                # Sorting by length ensures specific entities ("Blue Master") are matched before generic ones ("Blue").
            

                # This is ugly. Please don't kill me
                # Between constraints have a distance, which is a number, but we shouldn't confuse the distance with a value from our domain. 
                if ' between ' in con: 
                    for word in con.split(' '):
                        if word[:3].lower() in curr_domain_short and first==None:
                            first = curr_domain[curr_domain_short.index(word.lower()[:3])]
                        elif word[:3].lower() in curr_domain_short:
                            second = curr_domain[curr_domain_short.index(word.lower()[:3])]
                        
                        #permanently temporary
                        if word in word_num:
                            tmp_distance = word_num[word]
                #for every other constraint a number can also be a argument so we convert that first.
                else: 
                    for word in con.split(' '):
                        if word in word_num:
                            word = str(word_num[word])
                        
                        if word[:3].lower() in curr_domain_short and first==None:
                            first = curr_domain[curr_domain_short.index(word.lower()[:3])]
                        elif word[:3].lower() in curr_domain_short:
                            second = curr_domain[curr_domain_short.index(word.lower()[:3])]
                

                match(con):
                    #leftConstraint
                    #- clue contains 'left'
                    #- clue can be 'somewhere to the left' or 'directly left of' 
                    case _ if ' left' in con and ' directly ' in con:
                        #direct left constraint    
                        print(f'directly left: {con}')
                        print(LeftConstraint(first, second))
                        constraint_list.append(LeftConstraint(first, second))
                        
                    case _ if ' left' in con and ' somewhere ' in con:
                        #somewhere left constraint
                        print(f'somewhere left: {con}')
                        print(LeftConstraint(first, second, direct=False))
                        constraint_list.append(LeftConstraint(first, second, direct=False))

                    #rightConstraint same as left constraint
                    case _ if ' right' in con and ' somewhere ' in con: 
                        #somewhere right constraint
                        print(f'somewhere right: {con}')
                        print(RightConstraint(first, second, direct=False))
                        constraint_list.append(RightConstraint(first, second, direct=False))

                    case _ if ' right' in con and ' directly ' in con: 
                        #there shuoldn't be a 'directly right' constraint. Leaving it just in case.    
                        #direct right constraint
                        print(f'directly right: {con}')
                        print(RightConstraint(first, second))
                        constraint_list.append(RightConstraint(first, second))
                    
                    #nextToConstraint: 
                    case _ if ' next to each other' in con: 
                        print(f'next to each other: {con}')
                        print(LeftOrRightConstraint(first, second))
                        constraint_list.append(LeftOrRightConstraint(first, second))

                    #betweenConstraint
                    case _ if ' between' in con: 
                        print(f'between: {con}')
                        print(BetweenConstraint(first, second, distance=tmp_distance))
                        constraint_list.append(BetweenConstraint(first, second, distance=tmp_distance))
                
                    #isNotConstraint
                    case _ if ' is not ' in con: 
                        print(f'is not: {con}')
                        print(IsNotConstraint(first, second))
                        constraint_list.append(IsNotConstraint(first, second))

                    #isConstraint
                    case _: 
                        print(f'is: {con}')
                        print(IsConstraint(first, second))
                        constraint_list.append(IsConstraint(first, second))
            self.result.at[index, 'constraints'] = constraint_list
                

class Constraint: 
    """Parent Class for all Constraints"""
    def __init__(self, x, y):
        self.x = x
        self.y = y 

    def __repr__(self):
        return f'<{self.__class__.__name__}, {self.x}, {self.y}>'

    def is_satisfied(self):
        pass


class LeftConstraint(Constraint):
    """Constraint for 'x is left of y'"""
    def __init__(self, x, y, direct=True):
        super().__init__(x, y)
        self.direct=direct
    
    def is_satisfied(self):
        return super().is_satisfied()
    
    def __repr__(self):
        return f'<{self.__class__.__name__}, {self.x}, {self.y}, direct={self.direct}>' 
    
class IsConstraint(Constraint):
    """Constraint for 'x is y' """
    def __init__(self,x,y):
        super().__init__(x, y)

    def is_satisfied(self):
        return super().is_satisfied()
    
class IsNotConstraint(Constraint):
    """Constraint for 'x is not y' """
    def __init__(self,x,y):
        super().__init__(x, y)
    
    def is_satisfied(self):
        return super().is_satisfied()
    
class LeftOrRightConstraint(Constraint):
    """Constraint for 'x is next to y'"""
    def __init__(self, x, y, distance:int=0):
        super().__init__(x, y)
        self.distance=distance

    def is_satisfied(self):
        return super().is_satisfied()
    
class BetweenConstraint(Constraint):
    """Constraint for 'between x and y is distance House'"""
    def __init__(self,x,y,distance:int|None=0):
        super().__init__(x, y)
        self.distance=distance

    def is_satisfied(self):
        return super().is_satisfied()
    
    def __repr__(self):
        return f'<{self.__class__.__name__}, {self.x}, {self.y}, distance={self.distance}>' 

class RightConstraint(Constraint):
    """Constraint for 'x is right of y'"""
    def __init__(self, x, y, direct=True):
        super().__init__(x, y)
        self.direct=direct

    def is_satisfied(self):
        return super().is_satisfied()
    
if __name__ == "__main__":
    df = pd.read_parquet("data/Gridmode-00000-of-00001.parquet")
    dp = DataParsing(df)
    print(dp.get_csp())