# InitialValueProblems
## Introduction
This is the repo for the final exercise in Initial Value Problems.
There are some solutions of the first assignment. They won't work out of the box, because now the code is fit for the 2 dimensional problem of the final assignment.

## Modules
- `main.py` runs everything.
- `problems` includes all schemes (and their solutions)
  - `problem.py` is the root class for all schemes.
  - `second_order` includes all the schemes of the final exercise.
  - `second_order.py` has the common logic of all schemes in the final assignment
- `config` is a directory of all configuations. They are `toml` files with required fields. 
  - The modified FE requires also `sigma` configuration.
- `finite_differences.py` is the engine the multiplies the scheme's operator with the current state (and ticks the time).
- `fit.py` runs the error decay fit.
- `utils.py` mainly helps with construction of the differencial operators.

## How to run
To run the exprimemts, 

```
pip install -r requirements.txt
python main.py
```

Configure what to run with comment out in `main.py`, in the `run()` function.
