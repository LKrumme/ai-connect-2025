import copy
from typing import Dict, List, Set, Tuple, Any, Optional

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

        # Pre-calculate variable degree (number of constraints) for Heuristic
        self.var_degree = {v: 0 for v in variables}
        for c in constraints:
            if c.get('type') == 'alldiff':
                for v in c.get('variables', []):
                    if v in self.var_degree: self.var_degree[v] += len(c['variables']) - 1
            else:
                v1, v2 = c.get('var1'), c.get('var2')
                if v1 in self.var_degree: self.var_degree[v1] += 1
                if v2 in self.var_degree: self.var_degree[v2] += 1

    def reset(self):
        self.domains = copy.deepcopy(self.original_domains)
        self.assignment = {}
        self.search_steps = 0
        self.trace = []

    def is_complete(self):
        return len(self.assignment) == len(self.variables)

    def select_unassigned_variable(self):

        # MRV + Degree Heuristic:

        unassigned = [v for v in self.variables if v not in self.assignment]
        if not unassigned:
            return None

        # Sort by: 1. Domain Size (asc), 2. Degree (desc)
        return min(unassigned, key=lambda v: (len(self.domains[v]), -self.var_degree[v]))

    def order_domain_values(self, var):
        #Least Constraining Value (LCV) Heuristic.
        if len(self.assignment) == len(self.variables) - 1:
            return list(self.domains[var])

        def count_conflicts(value):
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

        return sorted(list(self.domains[var]), key=count_conflicts)

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
            assigned_in_constraint = [v for v in vars_in_constraint if v in assignment]
            if len(assigned_in_constraint) > 1:
                values = [assignment[v] for v in assigned_in_constraint]
                if len(values) != len(set(values)):
                    return False

        var1 = constraint.get('var1')
        var2 = constraint.get('var2')

        if var1 not in assignment or var2 not in assignment:
            return True

        val1, val2 = assignment[var1], assignment[var2]

        if ctype == 'equal': return val1 == val2
        elif ctype == 'not_equal': return val1 != val2
        elif ctype == 'adjacent': return abs(val1 - val2) == 1
        elif ctype == 'left_of': return val1 < val2
        elif ctype == 'right_of': return val1 > val2
        elif ctype == 'directly_left': return val1 == val2 - 1
        elif ctype == 'directly_right': return val1 == val2 + 1
        elif ctype == 'distance':
            dist = constraint.get('diff', 0)
            return abs(val1 - val2) == dist

        return True

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

            elif ctype in ['equal', 'not_equal', 'adjacent', 'left_of', 'right_of', 'directly_left', 'directly_right', 'distance']:
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
        """
        Prunes values that have NO chance of being part of a solution
        before we even start searching.
        """
        queue = []
        # Add all binary constraints to queue
        for c in self.constraints:
            ctype = c.get('type')
            if ctype not in ['alldiff'] and 'var1' in c and 'var2' in c:
                queue.append((c['var1'], c['var2'], c))
                queue.append((c['var2'], c['var1'], c))

        while queue:
            Xi, Xj, constraint = queue.pop(0)
            if self._revise(Xi, Xj, constraint):
                if len(self.domains[Xi]) == 0:
                    return False

                # If we removed values from Xi, we must re-check its neighbors
                for c2 in self.constraints:
                    if c2.get('type') == 'alldiff': continue
                    if 'var1' not in c2: continue

                    neighbor = None
                    if c2['var1'] == Xi and c2['var2'] != Xj: neighbor = c2['var2']
                    elif c2['var2'] == Xi and c2['var1'] != Xj: neighbor = c2['var1']

                    if neighbor:
                        queue.append((neighbor, Xi, c2))
        return True

    def _revise(self, Xi, Xj, constraint):
        revised = False
        to_remove = set()

        for x_val in self.domains[Xi]:
            # If theres ANY value in Xj that satisfies the constraint
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
        if self.search_steps > self.max_steps:
            return None

        if self.is_complete():
            return self.assignment.copy()

        var = self.select_unassigned_variable()
        if var is None: return None

        for value in self.order_domain_values(var):
            if self.is_consistent(var, value):
                self.assignment[var] = value
                removed = self.forward_check(var, value)

                if all(len(self.domains[v]) > 0 for v in self.variables if v not in self.assignment):
                    result = self.backtrack()
                    if result is not None: return result

                del self.assignment[var]
                self.restore_domains(removed)
        return None

    def solve(self):
        self.reset()
        if not self.ac3():
            return None # Impossible puzzle

        return self.backtrack()

    def get_search_steps(self):
        return self.search_steps