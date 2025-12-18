#--- data_parsing.py ---
import json

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
        for index, text in enumerate(self.df["puzzle"]):
            self.result.at[index, 'variables'] = ["Houses", "Names", "Colors", "Pets"]

    def _domains(self):
        # The current logic assumes the lists in the text are always in the same order
        # as the 'variables' list. In some puzzles, these are shuffled (e.g., Colors comes before Names).
        # We should use keywords (like checking if 'Red' is in the list) to match the correct list to the correct variable.
        # Otherwise, we end up assigning 'Red' to 'Name' and the solver breaks.

        text = self.df["puzzle"]

        for index, problem in enumerate(text):
            results = {}
            houses = []

            # Filling up Houses
            for h_Count in range(int(self.df["size"][index][0])):
                houses.append(h_Count)

            results["House"] = houses
            # Pattern looks for a Number, followed by a . (dot) at the start of the line and an optional space
            constraint_pattern = re.compile(r'^\d+\.\s*(\w+)')
            names = []
            colors = []
            pets = []

            for line in problem.splitlines():

                # Looking for Names
                match = constraint_pattern.match(line)
                if match:

                    first_word = match.group(1)

                    # The only possible first words in a Constraint can be "The", "House" and a Name, so we use this to filter out the Names
                    if (first_word != "The" and first_word != "House"):

                        if first_word not in names:
                            names.append(first_word)

                # Looking for Colors and Pets
                elif re.match(r'^Colors:\s*(.+?)\.?$', line):
                    colors = [c.strip() for c in re.match(r'^Colors:\s*(.+?)\.?$', line).group(1).split(',')]
                    results["Colors"] = colors

                elif re.match(r'^Pets:\s*(.+?)\.?$', line):
                    pets = [c.strip() for c in re.match(r'^Pets:\s*(.+?)\.?$', line).group(1).split(',')]
                    results["Pets"] = pets

            if len(names) < 3:
                for i in range(3 - len(names)):
                    names.append(f"")

            results["Names"] = names

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

                        numbers = re.findall(r'\d', word)
                        # match numbers
                        if numbers and first==None:
                            first = str(numbers[0])
                        if numbers:
                            second = str(numbers[0])

                match(con):
                    #leftConstraint
                    #- clue contains 'left'
                    #- clue can be 'somewhere to the left' or 'directly left of'
                    case _ if (' left' in con and ' directly ' in con) or (' left' in con and ' immediately ' in con):
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

                    case _ if (' right' in con and ' directly ' in con) or (' right' in con and ' immediately ' in con):
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
                    case _ if ' is not ' in con or ' does not ' in con:
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
    df = pd.read_parquet("data/Test_100_Puzzles.parquet")
    dp = DataParsing(df)
    print(dp.get_csp())



#--- bridge.py ---

import re
import copy

class BridgeParser:
    def __init__(self, size_str, constraints, domains):
        self.size_str = str(size_str)
        self.constraints = constraints
        self.domains = domains

        try:
            self.num_houses = int(self.size_str.split('*')[0])
        except:
            self.num_houses = 5

        self.item_to_var_map = {}
        self.var_to_domain_map = {}
        self._build_maps()

    def _build_maps(self):
        for category, items in self.domains.items():
            if category == "House": continue
            for item in items:
                clean_item = str(item).strip()
                var_name = f"{category}_{clean_item}"
                self.item_to_var_map[clean_item.lower()] = var_name
                self.var_to_domain_map[var_name] = set(range(1, self.num_houses + 1))

    def get_solver_data(self):
        variables = list(self.var_to_domain_map.keys())
        domains = copy.deepcopy(self.var_to_domain_map)
        final_constraints = []

        # Add AllDiff Constraints
        categories = {}
        for var in variables:
            cat = var.split('_')[0]
            if cat not in categories: categories[cat] = []
            categories[cat].append(var)

        for cat, vars_in_cat in categories.items():
            final_constraints.append({"type": "alldiff", "variables": vars_in_cat})

        for c in self.constraints:
            # Handle both Dictionary and Class formats
            if isinstance(c, dict):
                raw_x, raw_y = c.get('var1'), c.get('var2')
                ctype = c.get('type')
                direct = c.get('direct', False)
                is_anchor = isinstance(raw_y, int)
            else:
                raw_x = getattr(c, 'x', None)
                raw_y = getattr(c, 'y', None)
                ctype = type(c).__name__
                direct = getattr(c, 'direct', False)
                is_anchor = isinstance(raw_y, int)

            var1 = self.item_to_var_map.get(str(raw_x).lower())

            if is_anchor:
                var2 = raw_y # Keep as integer
            else:
                var2 = self.item_to_var_map.get(str(raw_y).lower())

            if var1:
                # If Anchor, Apply Domain Reduction
                if isinstance(var2, int):
                    if var1 in domains:
                        if ctype in ['IsNotConstraint', 'not_equal']:
                            #for example, "Guy is NOT 2" then Remove 2 from domain
                            domains[var1].discard(var2)
                        else:
                            # "Guy IS 2" then Keep ONLY 2 in domain
                            domains[var1] &= {var2}
                    continue

                    # If Binary, Add to Solver Constraints
                if var2:
                    # Normalize types
                    if ctype in ['LeftConstraint', 'left_of']:
                        final_constraints.append({"type": "left_of", "var1": var1, "var2": var2, "direct": direct})
                    elif ctype in ['RightConstraint', 'right_of']:
                        final_constraints.append({"type": "right_of", "var1": var1, "var2": var2, "direct": direct})
                    elif ctype in ['LeftOrRightConstraint', 'adjacent']:
                        final_constraints.append({"type": "adjacent", "var1": var1, "var2": var2})
                    elif ctype in ['IsConstraint', 'equal']:
                        final_constraints.append({"type": "equal", "var1": var1, "var2": var2})
                    elif ctype in ['IsNotConstraint', 'not_equal']:
                        final_constraints.append({"type": "not_equal", "var1": var1, "var2": var2})
                    elif ctype in ['BetweenConstraint', 'between_constraint']:
                        dist_val = getattr(c, 'distance', 0)
                        if dist_val is None: dist_val = 0
                        final_constraints.append({
                            "type": "distance",
                            "var1": var1,
                            "var2": var2,
                            "diff": int(dist_val) + 1
                        })

        return variables, domains, final_constraints



#--- generate_traces.py ---

def generate_traces(parquet_path, output_path="traces.json", limit=1000):
    df = pd.read_parquet(parquet_path)

    if limit:
        df = df.head(limit)

    parser = DataParsing(df)
    parsed_data = parser.get_csp()

    all_traces = {}

    for index, row in parsed_data.iterrows():
        try:
            puzzle_id = row['id']
            size_str = row['size']

            bridge = BridgeParser(size_str, row['constraints'], row['domains'])
            vars, doms, cons = bridge.get_solver_data()

            # We set a high max_steps because we want full traces for valid puzzles
            solver = Solver(vars, doms, cons, enable_trace=True, max_steps=10000)
            solution = solver.solve()

            if solution:
                # Store the successful trace
                all_traces[puzzle_id] = {
                    "status": "solved",
                    "steps": solver.search_steps,
                    "trace_data": solver.get_traces()
                }
                print(f"{puzzle_id}: Solved with trace ({len(solver.get_traces())} steps)")
            else:
                print(f"{puzzle_id}: Failed to solve")

        except Exception as e:
            print(f"Error on {row['id']}: {e}")

    # Save to file
    with open(output_path, "w") as f:
        json.dump(all_traces, f, indent=2)

#--- csp_solver.py ---

import copy
from typing import Dict, List, Set, Any, Tuple, Optional

class Solver:
    def __init__(self, variables: List[str], domains: Dict[str, Set[Any]],
                 constraints: List[Dict[str, Any]], enable_trace=False, max_steps=10000):
        self.variables = variables
        self.domains = {v: set(d) for v, d in domains.items()}
        self.original_domains = copy.deepcopy(self.domains)
        self.constraints = constraints
        self.assignment = {}
        self.search_steps = 0
        self.enable_trace = enable_trace
        self.trace = []
        self.max_steps = max_steps

        self.var_degree = {v: 0 for v in variables}
        for c in self.constraints:
            if c.get('type') == 'alldiff':
                # Fully connected graph within alldiff group
                degree_increase = len(c['variables']) - 1
                for v in c.get('variables', []):
                    if v in self.var_degree: self.var_degree[v] += degree_increase
            else:
                # Binary constraints
                for v_key in ['var1', 'var2']:
                    v = c.get(v_key)
                    if v in self.var_degree: self.var_degree[v] += 1

    def reset(self):
        self.domains = copy.deepcopy(self.original_domains)
        self.assignment = {}
        self.search_steps = 0
        self.trace = []

    def is_complete(self):
        return len(self.assignment) == len(self.variables)

    def select_unassigned_variable(self):
        # MRV + Degree Heuristic
        unassigned = [v for v in self.variables if v not in self.assignment]
        if not unassigned: return None
        return min(unassigned, key=lambda v: (len(self.domains[v]), -self.var_degree[v]))

    def order_domain_values(self, var):
        # LCV Heuristic
        if len(self.assignment) == len(self.variables) - 1:
            return list(self.domains[var])

        return sorted(list(self.domains[var]), key=lambda val: self._count_conflicts(var, val))

    def _count_conflicts(self, var, value):
        #Helper to count how many options this value eliminates for neighbors
        count = 0
        for constraint in self.constraints:
            if constraint.get('type') == 'alldiff': continue

            neighbor = None
            if constraint.get('var1') == var: neighbor = constraint.get('var2')
            elif constraint.get('var2') == var: neighbor = constraint.get('var1')

            if neighbor and neighbor not in self.assignment:
                for n_val in self.domains[neighbor]:
                    test_assign = {var: value, neighbor: n_val}
                    if not self._check_constraint(constraint, test_assign):
                        count += 1
        return count

    def is_consistent(self, var, value):
        test_assignment = self.assignment.copy()
        test_assignment[var] = value
        for constraint in self.constraints:
            if not self._check_constraint(constraint, test_assignment):
                return False
        return True

    def _check_constraint(self, constraint: Dict, assignment: Dict) -> bool:
        ctype = constraint.get('type')

        if ctype == 'alldiff':
            vars_in_constraint = constraint['variables']
            assigned_values = [assignment[v] for v in vars_in_constraint if v in assignment]
            return len(assigned_values) == len(set(assigned_values))

        var1, var2 = constraint.get('var1'), constraint.get('var2')
        if var1 not in assignment or var2 not in assignment:
            return True

        val1, val2 = assignment[var1], assignment[var2]

        match ctype:
            case 'equal': return val1 == val2
            case 'not_equal': return val1 != val2
            case 'adjacent': return abs(val1 - val2) == 1

            case 'left_of':
                if constraint.get('direct') is True:
                    return val1 == val2 - 1  # Directly Left
                return val1 < val2           # Somewhere Left

            case 'right_of':
                if constraint.get('direct') is True:
                    return val1 == val2 + 1  # Directly Right
                return val1 > val2           # Somewhere Right

            case 'distance': return abs(val1 - val2) == constraint.get('diff', 0)
            case _: return True

    def forward_check(self, var, value):
        removed = {}
        for constraint in self.constraints:
            ctype = constraint.get('type')

            if ctype == 'alldiff':
                vars_in_constraint = constraint['variables']
                if var in vars_in_constraint:
                    for other_var in vars_in_constraint:
                        if other_var != var and other_var not in self.assignment:
                            if value in self.domains[other_var]:
                                if other_var not in removed: removed[other_var] = set()
                                removed[other_var].add(value)
                                self.domains[other_var].discard(value)

            # Logic for Binary Constraints
            elif 'var1' in constraint and 'var2' in constraint:
                var1, var2 = constraint.get('var1'), constraint.get('var2')
                other_var = None
                if var1 == var and var2 not in self.assignment: other_var = var2
                elif var2 == var and var1 not in self.assignment: other_var = var1

                if other_var:
                    to_remove = set()
                    for other_val in list(self.domains[other_var]):
                        test_assign = {var: value, other_var: other_val}
                        if not self._check_constraint(constraint, test_assign):
                            to_remove.add(other_val)
                    if to_remove:
                        if other_var not in removed: removed[other_var] = set()
                        removed[other_var].update(to_remove)
                        self.domains[other_var] -= to_remove
        return removed

    def restore_domains(self, removed):
        for var, values in removed.items():
            self.domains[var].update(values)

    def ac3(self):
        queue = []
        for c in self.constraints:
            if c.get('type') != 'alldiff' and 'var1' in c and 'var2' in c:
                queue.append((c['var1'], c['var2'], c))
                queue.append((c['var2'], c['var1'], c))

        while queue:
            Xi, Xj, constraint = queue.pop(0)
            if self._revise(Xi, Xj, constraint):
                if len(self.domains[Xi]) == 0: return False
                for c2 in self.constraints:
                    if c2.get('type') == 'alldiff': continue
                    if 'var1' not in c2: continue

                    neighbor = None
                    if c2['var1'] == Xi and c2['var2'] != Xj: neighbor = c2['var2']
                    elif c2['var2'] == Xi and c2['var1'] != Xj: neighbor = c2['var1']

                    if neighbor: queue.append((neighbor, Xi, c2))
        return True

    def _revise(self, Xi, Xj, constraint):
        revised = False
        to_remove = set()
        for x_val in self.domains[Xi]:
            satisfiable = False
            for y_val in self.domains[Xj]:
                test_assign = {Xi: x_val, Xj: y_val}
                if self._check_constraint(constraint, test_assign):
                    satisfiable = True
                    break
            if not satisfiable:
                to_remove.add(x_val)
                revised = True
        self.domains[Xi] -= to_remove
        return revised

    def backtrack(self):
        self.search_steps += 1
        if self.search_steps > self.max_steps: return None
        if self.is_complete(): return self.assignment.copy()

        var = self.select_unassigned_variable()
        if var is None: return None

        if self.enable_trace:
            self.trace.append({
                "step": self.search_steps,
                "current_assignment": self.assignment.copy(),
                "domain_sizes": {v: len(self.domains[v]) for v in self.variables},
                "chosen_variable": var,
                "action": "branch"
            })

        for value in self.order_domain_values(var):
            if self.is_consistent(var, value):
                self.assignment[var] = value
                removed = self.forward_check(var, value)

                result = self.backtrack()
                if result is not None: return result

                del self.assignment[var]
                self.restore_domains(removed)
        return None

    def solve(self) -> Tuple[Optional[Dict], str]:
        """
        Status: 'SOLVED', 'IMPOSSIBLE', 'TIMEOUT'
        """
        self.reset()
        if not self.ac3():
            return None, "IMPOSSIBLE (AC-3 Wiped Domain)" #parser likely picked up a "False Positive" constraint that directly conflicted with another

        result = self.backtrack()

        if result:
            return result, "SOLVED"
        elif self.search_steps > self.max_steps:
            return None, "TIMEOUT (Step Limit Reached)"
        else:
            return None, "IMPOSSIBLE (Search Exhausted)"

    def get_traces(self):
        return self.trace
