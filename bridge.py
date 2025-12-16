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