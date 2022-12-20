### Project Structure

```
final-project
├─ .DS_Store
├─ README.md
├─ VRPTW
│  ├─ .DS_Store
│  ├─ __init__.py
│  ├─ parser.py
│  ├─ solvers
│  │  └─ algorithms.py
│  └─ structure.py
├─ draw_problem.ipynb
├─ instances
│  ├─ C101.txt
│  ├─ C108.txt
│  ├─ C266.txt
│  ├─ R202.txt
│  └─ RC207.txt
└─ main.py

```

### Running the Project

Command to run the script: `python3 main.py instances/C101.txt`

Parameters of the script:

- Solomon dataset file location
  - Examples: instances/C101.txt , instances/C108.txt , instances/C266.txt etc...

### Algorithms used in this project, to solve the VRPTW (Vehicle Routing Problem with Time Windows)

- Iterated local search (ILS)

- Genetic Algorithm (GA)

### Data

- All input data is in [Solomon benchmark format](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/documentation/)

### Output

Once the script run is complete, it shows distance of optimal route and time taken to calculate it for both algortihms: ILS and GA
