import pandas as pd
import json
import time
from data_parsing import DataParsing
from bridge import BridgeParser
from solver import Solver

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

def evaluate_dataset(parquet_path, output_path="final_results.json", limit=None):
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

    print("-" * 50)

    for index, row in parsed_data.iterrows():
        p_id = row['id']
        try:
            bridge = BridgeParser(row['size'], row['constraints'], row['domains'])
            vars, doms, cons = bridge.get_solver_data()

            # Sanity Check for Empty Parser
            if len(cons) < 1:
                print(f"Puzzle {index+1}: Parser found only {len(cons)} constraints")
                results["failed"] += 1
                continue

            solver = Solver(vars, doms, cons, max_steps=5000)
            solution, status = solver.solve()

            steps = solver.search_steps
            results["total_steps"] += steps

            if solution:
                ground_truth = df.iloc[index]['solution']
                is_correct, reason = verify_ground_truth(solution, ground_truth, bridge)

                if is_correct:
                    results["solved"] += 1
                    results["results"][p_id] = {
                        "status": "solved",
                        "steps": steps,
                        "verified": True
                    }
                    print(f"✓ Puzzle {index+1}: Solved & Verified ({steps} steps)")
                else:
                    results["failed"] += 1
                    results["results"][p_id] = {
                        "status": "failed",
                        "reason": f"Wrong Solution: {reason}",
                        "steps": steps
                    }
                    print(f"Puzzle {index+1}: Solved but WRONG -> {reason}")
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

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to '{output_path}'")

if __name__ == "__main__":
    PATH = "data/Gridmode-00000-of-00001.parquet"
    evaluate_dataset(PATH)