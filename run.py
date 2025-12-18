import pandas as pd
import json
import time
from solver import DataParsing
from solver import BridgeParser
from solver import generate_traces
from solver import Solver

def format_grid_solution(assignment, size):
    if not assignment: return "{}"

    raw_categories = list(set(k.split('_')[0] for k in assignment.keys() if '_' in k))

    # matches the prof's sequence
    priority_order = ["name", "nationality", "color", "pet", "drink", "cigarette", "smoke", "job", "profession"]

    def category_sort_key(cat):
        # Normalize category to match the priority list (lowercase, remove plural 's')
        norm = cat.lower()
        if norm.endswith('ies'): norm = norm[:-3] + 'y'
        elif norm.endswith('s') and not norm.endswith('ss'): norm = norm[:-1]

        if norm in priority_order:
            return priority_order.index(norm)
        return 999

    categories = sorted(raw_categories, key=category_sort_key)

    display_header = ["House"]
    for cat in categories:
        if cat.endswith('ies'):
            display_header.append(cat[:-3] + 'y')
        elif cat.endswith('s') and not cat.endswith('ss'):
            display_header.append(cat[:-1])
        else:
            display_header.append(cat)

    rows = [[str(i+1)] + [""] * len(categories) for i in range(size)]

    for key, house_num in assignment.items():
        if '_' in key:
            category, value = key.split('_', 1)

            if category in categories:
                col_idx = categories.index(category) + 1
                row_idx = int(house_num) - 1

                if 0 <= row_idx < size:
                    rows[row_idx][col_idx] = value

    return json.dumps({"header": display_header, "rows": rows})

def verify_ground_truth(solver_assignments, ground_truth_raw, bridge):
    if not solver_assignments:
        return False, "No Solution"

    try:
        data = json.loads(ground_truth_raw) if isinstance(ground_truth_raw, str) else ground_truth_raw
    except:
        return False, "JSON Error"

    headers = data.get("header", [])
    rows = data.get("rows", [])

    if len(rows) == 0 or len(headers) == 0:
        return False, "Empty Data"

    if hasattr(rows, 'tolist'):
        rows = rows.tolist()

    if len(rows) > 0 and hasattr(rows[0], 'tolist'):
        rows = [r.tolist() for r in rows]

    if len(rows) > 0 and not isinstance(rows[0], (list, tuple)):
        try:
            width = len(headers)
            rows = [rows[i : i + width] for i in range(0, len(rows), width)]
        except:
            return False, "Malformed Data"

    # Verification loop
    for row in rows:
        try:
            if len(row) == 0: continue
            true_house = int(row[0])
        except (ValueError, IndexError, TypeError):
            continue

        for col_idx, val in enumerate(row):
            if col_idx == 0: continue
            if col_idx >= len(headers): break

            var_key = bridge.item_to_var_map.get(str(val).lower())

            if var_key:
                if var_key not in solver_assignments:
                    return False, f"Missing: {var_key}"

                if solver_assignments[var_key] != true_house:
                    return False, f"Mismatch: {var_key} ({solver_assignments[var_key]} vs {true_house})"

    return True, "Match"

def evaluate_dataset(parquet_path, csv_path="result.csv", limit=None):
    print(f"Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if limit:
        print(f"Limiting run to first {limit} puzzles.")
        df = df.head(limit)

    start_parse = time.time()
    parser = DataParsing(df)
    parsed_data = parser.get_csp()
    print(f"Parsing completed in {time.time() - start_parse:.2f}s")

    results = {
        "total": len(df),
        "solved": 0,
        "failed": 0,
        "errors": 0,
        "total_steps": 0,
        "results": {}
    }

    csv_results = []

    print("-" * 50)

    for index, row in parsed_data.iterrows():
        p_id = row['id']
        puzzle_size = int(row['size'][0]) # Assuming 'size' is like ['5', '5']

        try:
            bridge = BridgeParser(row['size'], row['constraints'], row['domains'])
            vars, doms, cons = bridge.get_solver_data()

            # Sanity Check for Empty Parser
            if len(cons) < 1:
                print(f"Puzzle {index+1}: Parser found only {len(cons)} constraints")
                results["failed"] += 1
                # Add empty result to CSV for consistency
                csv_results.append({"id": p_id, "grid_solution": "{}", "steps": 0})
                continue

            solver = Solver(vars, doms, cons, max_steps=5000)
            solution, status = solver.solve()

            steps = solver.search_steps
            results["total_steps"] += steps

            if solution:
                grid_json = format_grid_solution(solution, puzzle_size)
            else:
                grid_json = "{}" # Empty string/json for failed puzzles

            csv_results.append({
                "id": p_id,
                "grid_solution": grid_json,
                "steps": steps
            })

            if solution:
                # Handling both of the puzzle files
                # for the Gridmode file, with the solution column
                if 'solution' in df.columns:
                    ground_truth = df.iloc[index]['solution']
                    is_correct, reason = verify_ground_truth(solution, ground_truth, bridge)

                    if is_correct:
                        results["solved"] += 1
                        results["results"][p_id] = {"status": "solved", "steps": steps, "verified": True}
                        print(f"✓ Puzzle {index+1}: Solved & Verified ({steps} steps)")
                    else:
                        results["failed"] += 1
                        results["results"][p_id] = {"status": "failed", "reason": reason, "steps": steps}
                        print(f"Puzzle {index+1}: Solved but WRONG -> {reason}")

                # for the Test_100_Puzzles file, as it has no solution column
                else:
                    results["solved"] += 1
                    results["results"][p_id] = {"status": "submitted", "steps": steps}
                    print(f"? Puzzle {index+1}: Solved")
            else:
                results["failed"] += 1
                results["results"][p_id] = {
                    "status": "failed",
                    "reason": status,
                    "steps": steps
                }
                print(f"✗ Puzzle {index+1}: Failed -> {status}")

        except Exception as e:
            results["errors"] += 1
            results["results"][p_id] = {"status": "error", "error": str(e)}
            # Add error result to CSV
            csv_results.append({"id": p_id, "grid_solution": "{}", "steps": 0})
            print(f"Error on {p_id}: {e}")

    # Final Stats
    print("=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Total Puzzles: {results['total']}")
    success_rate = (results['solved'] / results['total']) * 100
    avg_steps = results['total_steps'] / results['total'] if results['total'] > 0 else 0

    print(f"Success Rate:  {success_rate:.2f}% ({results['solved']}/{results['total']})")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Failures:      {results['failed']}")
    print(f"Errors:        {results['errors']}")
    print("=" * 50)

    # Save Submission CSV
    pd.DataFrame(csv_results).to_csv(csv_path, index=False)
    print(f"Submission file saved to '{csv_path}'")

if __name__ == "__main__":
    PATH = "data/Test_100_Puzzles.parquet"
    generate_traces(PATH)
    evaluate_dataset(PATH)