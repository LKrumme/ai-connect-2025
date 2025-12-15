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
            case 'left_of': return val1 < val2
            case 'right_of': return val1 > val2
            case 'directly_left': return val1 == val2 - 1
            case 'directly_right': return val1 == val2 + 1
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