import pandas as pd
import json
import time
from data_parsing import DataParsing
from bridge import BridgeParser
from solver import Solver

def evaluate_dataset(parquet_path, output_path="final_results.json", limit=None):
    print(f"Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if limit:
        print(f"Limiting run to first {limit} puzzles.")
        df = df.head(limit)

    print("Parsing puzzles (Teammate's Module)...")
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

            solver = Solver(vars, doms, cons, max_steps=5000)
            solution, status = solver.solve()

            steps = solver.search_steps
            results["total_steps"] += steps

            if solution:
                results["solved"] += 1
                results["results"][p_id] = {
                    "status": "solved",
                    "steps": steps
                }
                print(f"✓ Puzzle {index+1}: Solved ({steps} steps)")
            else:
                results["failed"] += 1
                results["results"][p_id] = {
                    "status": "failed",
                    "reason": status,
                    "steps": steps
                }

                print(f"✗ Puzzle {index+1}: Failed -> {status}")
                if index < 5:
                    print(f"  DEBUG: Vars: {len(vars)} | Constraints: {len(cons)}")
                    print(f"  Domains: {doms}")


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