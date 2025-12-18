import pandas as pd
import json
from data_parsing import DataParsing
from bridge import BridgeParser
from csp_solver import Solver

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

if __name__ == "__main__":
    PATH = "data/Test_100_Puzzles.parquet"
    generate_traces(PATH)