class BridgeParser:
    # Adapts raw parser output to solver
    def __init__(self, size_str, raw_constraints, raw_domains):
        # Extract grid size
        self.num_houses = int(str(size_str).split('*')[0])
        self.raw_constraints = raw_constraints
        self.raw_domains = raw_domains

        self.solver_vars = []
        self.solver_domains = {}
        self.solver_constraints = []
        # Map item names to solver variable names
        self.item_to_var_map = {}

    def get_solver_data(self):
        self._generate_variables()
        self._serialize_constraints()
        return self.solver_vars, self.solver_domains, self.solver_constraints

    def _generate_variables(self):
        # Flatten categories into individual variables
        for category, items in self.raw_domains.items():
            if not items or category == 'House':
                continue

            group_vars = []
            for item in items:
                # Create variable name like "Color_Red"
                var_name = f"{category}_{item}"
                self.solver_vars.append(var_name)
                # Assign domain
                self.solver_domains[var_name] = set(range(1, self.num_houses + 1))

                # Storing mapping for later lookup
                self.item_to_var_map[str(item).lower()] = var_name
                group_vars.append(var_name)

            # Ensure items in the same category are distinct
            self.solver_constraints.append({'type': 'alldiff', 'variables': group_vars})

    def _serialize_constraints(self):
        if not self.raw_constraints: return

        for c in self.raw_constraints:
            try:
                c_type = c.__class__.__name__

                # Handle Anchor constraints
                if 'PositionConstraint' in c_type:
                    raw_x = getattr(c, 'x', None)
                    pos = getattr(c, 'pos', None)

                    var1 = self.item_to_var_map.get(str(raw_x).lower())

                    if var1 and pos:
                        if pos <= self.num_houses:
                            self.solver_domains[var1] = {int(pos)}
                    continue

                # Handle Binary Constraints
                raw_x = getattr(c, 'x', None)
                raw_y = getattr(c, 'y', None)

                # Look up solver variable names
                var1 = self.item_to_var_map.get(str(raw_x).lower())
                var2 = self.item_to_var_map.get(str(raw_y).lower())

                if not var1 or not var2: continue

                entry = {}
                # Map constraint types to solver logic
                if 'LeftConstraint' in c_type:
                    direct = getattr(c, 'direct', False)
                    entry = {'type': 'directly_left' if direct else 'left_of', 'var1': var1, 'var2': var2}

                elif 'RightConstraint' in c_type:
                    direct = getattr(c, 'direct', False)
                    entry = {'type': 'directly_right' if direct else 'right_of', 'var1': var1, 'var2': var2}

                elif 'LeftOrRightConstraint' in c_type:
                    entry = {'type': 'adjacent', 'var1': var1, 'var2': var2}

                elif 'IsConstraint' in c_type or 'IsContraint' in c_type:
                    entry = {'type': 'equal', 'var1': var1, 'var2': var2}

                elif 'IsNotConstraint' in c_type:
                    entry = {'type': 'not_equal', 'var1': var1, 'var2': var2}

                elif 'Between' in c_type:
                    dist = getattr(c, 'distance', None)
                    # "Between" usually implies a gap of 1 house
                    diff_val = int(dist) + 1 if dist else 2
                    entry = {'type': 'distance', 'var1': var1, 'var2': var2, 'diff': diff_val}

                if entry: self.solver_constraints.append(entry)

            except Exception as e:
                pass