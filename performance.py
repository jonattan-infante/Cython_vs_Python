# Import Logistic Regression
from src.algorithms_python.LogisticRegression import training_model as training_model_python
from src.algorithms_cython.LogisticRegression import training_model as training_model_cython
# Import BackPack Genetic
from src.algorithms_python.BackpackGenetics import algorithm_genetic as algorithm_genetic_python
from src.algorithms_cython.BackpackGenetics import algorithm_genetic as algorithm_genetic_cython
import time

for i in range(30):
    init_time = time.time()
    training_model_cython()
    fin_time = time.time()
    time_cython = fin_time - init_time

    init_time = time.time()
    training_model_python()
    fin_time = time.time()
    time_python = fin_time - init_time

    with open("results/LogisticRegression.csv", "a") as archivo:
        archivo.write(f'{i},{time_python},{time_cython} \n')

for i in range(30):
    init_time = time.time()
    algorithm_genetic_cython(100, 1000)
    fin_time = time.time()
    time_cython = fin_time - init_time

    init_time = time.time()
    algorithm_genetic_python(100, 1000)
    fin_time = time.time()
    time_python = fin_time - init_time

    with open("results/BackPackGenetics.csv", "a") as archivo:
        archivo.write(f'{i},{time_python},{time_cython} \n')
