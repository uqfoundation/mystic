from mystic.solvers import DifferentialEvolutionSolver
from mystic.solvers import NelderMeadSimplexSolver, PowellDirectionalSolver
from mystic.termination import VTR
from mystic.models import rosen
from mystic.solvers import LoadSolver
import os

solver = PowellDirectionalSolver(3)
solver.SetRandomInitialPoints([0.,0.,0.],[10.,10.,10.])
term = VTR()
solver.Solve(rosen, term)
x = solver.bestSolution
y = solver.bestEnergy  
assert solver._state == None
assert LoadSolver(solver._state) == None

solver = PowellDirectionalSolver(3)
solver.SetRandomInitialPoints([0.,0.,0.],[10.,10.,10.])
term = VTR()
tmpfile = 'mysolver.pkl'
solver.SetSaveFrequency(10, tmpfile)
solver.Solve(rosen, term)
x = solver.bestSolution
y = solver.bestEnergy  
_solver = LoadSolver(tmpfile)
os.remove(tmpfile)
assert all(x == _solver.bestSolution)
assert y == _solver.bestEnergy  

solver = NelderMeadSimplexSolver(3)
solver.SetRandomInitialPoints([0.,0.,0.],[10.,10.,10.])
term = VTR()
solver.Solve(rosen, term)
x = solver.bestSolution
y = solver.bestEnergy  
assert solver._state == None
assert LoadSolver(solver._state) == None

solver = NelderMeadSimplexSolver(3)
solver.SetRandomInitialPoints([0.,0.,0.],[10.,10.,10.])
term = VTR()
solver.SetSaveFrequency(10, tmpfile)
solver.Solve(rosen, term)
x = solver.bestSolution
y = solver.bestEnergy  
_solver = LoadSolver(tmpfile)
os.remove(tmpfile)
assert all(x == _solver.bestSolution)
assert y == _solver.bestEnergy  

solver = DifferentialEvolutionSolver(3,40)
solver.SetRandomInitialPoints([0.,0.,0.],[10.,10.,10.])
term = VTR()
solver.Solve(rosen, term)
x = solver.bestSolution
y = solver.bestEnergy  
assert solver._state == None
assert LoadSolver(solver._state) == None

solver = DifferentialEvolutionSolver(3,40)
solver.SetRandomInitialPoints([0.,0.,0.],[10.,10.,10.])
term = VTR()
solver.SetSaveFrequency(10, tmpfile)
solver.Solve(rosen, term)
x = solver.bestSolution
y = solver.bestEnergy  
_solver = LoadSolver(tmpfile)
os.remove(tmpfile)
assert all(x == _solver.bestSolution)
assert y == _solver.bestEnergy  

solver = DifferentialEvolutionSolver(3,40)
solver.SetRandomInitialPoints([0.,0.,0.],[10.,10.,10.])
term = VTR()
solver.SetSaveFrequency(0, tmpfile)
solver.Solve(rosen, term)
x = solver.bestSolution
y = solver.bestEnergy  
_solver = LoadSolver(tmpfile)
os.remove(tmpfile)
assert all(x == _solver.bestSolution)
assert y == _solver.bestEnergy  

solver = DifferentialEvolutionSolver(3,40)
solver.SetRandomInitialPoints([0.,0.,0.],[10.,10.,10.])
term = VTR()
solver.SetSaveFrequency(None, tmpfile)
solver.Solve(rosen, term)
x = solver.bestSolution
y = solver.bestEnergy  
_solver = LoadSolver(tmpfile)
os.remove(tmpfile)
assert all(x == _solver.bestSolution)
assert y == _solver.bestEnergy  

solver = DifferentialEvolutionSolver(3,40)
solver.SetRandomInitialPoints([0.,0.,0.],[10.,10.,10.])
term = VTR()
solver.SetSaveFrequency(100000, tmpfile)
solver.Solve(rosen, term)
x = solver.bestSolution
y = solver.bestEnergy  
_solver = LoadSolver(tmpfile)
os.remove(tmpfile)
assert all(x == _solver.bestSolution)
assert y == _solver.bestEnergy  

solver = DifferentialEvolutionSolver(3,40)
solver.SetRandomInitialPoints([0.,0.,0.],[10.,10.,10.])
solver.SetSaveFrequency(100000)
term = VTR()
solver.Solve(rosen, term)
x = solver.bestSolution
y = solver.bestEnergy  
tmpfile = solver._state
_solver = LoadSolver(tmpfile)
os.remove(tmpfile)
assert all(x == _solver.bestSolution)
assert y == _solver.bestEnergy  

solver = DifferentialEvolutionSolver(3,40)
solver.SetRandomInitialPoints([0.,0.,0.],[10.,10.,10.])
term = VTR()
solver.SetSaveFrequency(0)
solver.Solve(rosen, term)
x = solver.bestSolution
y = solver.bestEnergy  
assert solver._state == None
assert LoadSolver(solver._state) == None

# EOF
