import numpy as np


class BackPack:
    def __init__(self, gen):
        self.gen = gen
        self.weight = [3, 7, 4, 1, 5, 4, 2, 3]
        self.fitness = sum(self.gen * self.weight)


def setting(init_population):
    return [BackPack(np.random.randint(2, size=8)) for i in range(init_population)]


def average_population_assessment(population):
    return np.mean([i.fitness for i in population])


def max_population_assessment(population):
    index = np.argmax([i.fitness for i in population])
    return population[index]


def select_parents(population):
    total_point = sum([i.fitness for i in population])
    probability = [i.fitness / total_point for i in population]
    parents = np.random.choice(population, p=probability, size=int(len(population) / 2) + 10, replace=False)
    return parents


def mutation(gen):
    gen[len(gen) - 1] = np.random.randint(2)
    gen = np.array(gen)
    return gen


def next_generation(fathers):
    new_generation = []
    for i in range(len(fathers) - 1):
        father_one_part_one, father_one_part_two = np.array_split(fathers[i].gen, 2)
        father_two_part_one, father_two_part_two = np.array_split(fathers[i + 1].gen, 2)
        son_one = BackPack(mutation([*father_one_part_one, *father_two_part_two]))
        son_two = BackPack(mutation([*father_one_part_two, *father_two_part_one]))
        new_generation.append(son_one)
        new_generation.append(son_two)
    return new_generation


def algorithm_genetic(generation, number_population):
    population = setting(number_population)
    metrics_by_generation = [average_population_assessment(population)]
    for i in range(generation):
        fathers = select_parents(population)
        population = next_generation(fathers)
        metrics_by_generation.append(average_population_assessment(population))
    best_gen = max_population_assessment(population)
    return metrics_by_generation, best_gen
