import pandas as pd
import time
import json
from data_parsing import DataParsing
from bridge import BridgeParser
from solver import Solver

def evaluate_dataset(file_path, limit=None):

    print(f"Loading dataset from: {file_path}")
    df = pd.read_parquet(file_path)

    if limit:
        df = df.head(limit)
        print(f"Limiting run to first {limit} puzzles.")

    print(f"Parsing {len(df)} puzzles (Teammate's Module)...")
    start_time = time.time()

    parser = DataParsing(df)
    parsed_data = parser.get_csp()

    print(f"Parsing completed in {time.time() - start_time:.2f}s")
    print("-" * 50)

    # Statistics
    stats = {
        'total': len(parsed_data),
        'solved': 0,
        'failed': 0,
        'errors': 0, # Parser or Bridge failures
        'total_steps': 0,
        'results': {}
    }

    # Loop through every puzzle
    for index, row in parsed_data.iterrows():
        p_id = row.get('id', f'index_{index}')

        try:
            if not isinstance(row['constraints'], list) or not isinstance(row['domains'], dict):
                raise ValueError("Invalid format from DataParsing module")

            bridge = BridgeParser(
                size_str=row['size'],
                raw_constraints=row['constraints'],
                raw_domains=row['domains']
            )
            vars, doms, cons = bridge.get_solver_data()

            # 4. Solver
            solver = Solver(vars, doms, cons)
            solution = solver.solve()

            steps = solver.get_search_steps()
            stats['total_steps'] += steps

            if solution:
                stats['solved'] += 1
                stats['results'][p_id] = {'status': 'solved', 'steps': steps}
                if index % 10 == 0:
                    print(f"✓ Puzzle {index+1}: Solved ({steps} steps)")
            else:
                stats['failed'] += 1
                stats['results'][p_id] = {'status': 'failed', 'steps': steps}
                print(f"✗ Puzzle {index+1}: Failed to find solution")

        except Exception as e:
            stats['errors'] += 1
            stats['results'][p_id] = {'status': 'error', 'error': str(e)}


    print("=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)

    accuracy = (stats['solved'] / stats['total']) * 100
    avg_steps = stats['total_steps'] / stats['solved'] if stats['solved'] > 0 else 0

    print(f"Total Puzzles: {stats['total']}")
    print(f"Success Rate:  {accuracy:.2f}% ({stats['solved']}/{stats['total']})")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Failures:      {stats['failed']} (Solver returned None)")
    print(f"Errors:        {stats['errors']} (Parser/Bridge crashed)")
    print("=" * 50)

    # Save details to JSON
    with open("final_results.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Detailed results saved to 'final_results.json'")

if __name__ == "__main__":
    PATH = "data/Gridmode-00000-of-00001.parquet"

    evaluate_dataset(PATH)