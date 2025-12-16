# ai-connect-2025

## Project Overview  
In this project, we have developed an intelligent system that solves complex logic puzzles automatically. In these tasks, often known as "Einstein Puzzles" or "Logic Grid Puzzles", in which a given Number of Houses, posess various variables (such as people, house colors, or professions), these must be correctly arranged to said houses based on text-based clues. A typical clue might read: "The owner of the green house lives directly to the left of the Dane."

While humans often require a long time to solve such tasks, our algorithm handles them in fractions of a second with mathematical precision.

We utilize a classic Constraint Satisfaction Problem (CSP) Solver.

---

### The four phases:

1. **Parsing:** First, our system reads the puzzle texts. The DataParsing module identifies key terms (e.g., "houses," "colors") as well as relationships/constraints ("left of", "next to", "is not", "in Between", ...).

2. **Solving:** Our Solver operates as a basic CSP Solver, it operates on the following principles:   
**Logical Reduction:** Before the system even starts trying out possibilities, it eliminates all options that contradict the constaints (process of elimination).  
**Strategic Selection:** The program always starts at the "most volatile" part of the puzzle (where there are the fewest possible choices to make) to keep complexity low.  
**Backtracking:** If a dead end is reached, the algorithm systematically takes one step back and tests alternative paths—similar to a pathfinder in a maze.

3. Tracing: Next up, we Trace the Process of our Solver by Noting, which decisions were made in what Situation

4. Evaluation: Lastly, we run an Evaluation of our solved Puzzles and Evaluate our Accuracy, Efficiency and Generalization

---

## Usage Guide  
// TODO: We don't have a run.py file Yet, will update when we do

