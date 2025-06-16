import numpy as np

import neat

generations_counter = 0

def initial_population(state, stats, config, output, fitness):
    population = neat.population.Population(config, [], [], state)
    if output:
        population.add_reporter(neat.reporting.StdOutReporter(True))
        population.add_reporter(neat.reporting.CSVReporter(fitness))
    population.add_reporter(stats)
    return population


def process_action(actions):
    action = np.argmax(actions)
    return action