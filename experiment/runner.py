import os
import pickle
import time
import sys
import numpy as np

from base_runner import BaseAttentionRunnerModule
from cma_es import CMAEvolutionStrategy
from experiment.configs.config import *
from neat.nn.recurrent import RecurrentNetwork
from parallel import ParallelEvaluator
from self_attention import SelfAttention
from utility import process_action, initial_population


class AttentionNEATModule(BaseAttentionRunnerModule):
    def __init__(self):
        super(AttentionNEATModule, self).__init__()

        self.attention_model = SelfAttention(input_shape=SelfAttentionConfig.IMAGE_SHAPE,
                                             patch_size=SelfAttentionConfig.PATCH_SIZE,
                                             patch_stride=SelfAttentionConfig.PATCH_STRIDE,
                                             transformer_d=SelfAttentionConfig.TRANSFORMER_D,
                                             top_k=SelfAttentionConfig.TOP_K,
                                             direction=BASE_DIR)

        self._layers.extend(self.attention_model.layers)

        self.cmaes_model = CMAEvolutionStrategy(population_size=CMAESConfig.POP_SIZE,
                                                init_sigma=CMAESConfig.INIT_SIGMA,
                                                init_params=self.get_params())

        __stats = neat.statistics.StatisticsReporter()
        self.population = initial_population(None, __stats, AttentionNEATConfig.NEAT_CONFIG, True)


def get_action(net, ob):
    # top = runner.attention_model.get_output(ob)
    # new_ob = runner.attention_model.normalize_patch_centers(top)
    # new_ob = np.append(new_ob, [1.0])
    # print(ob.shape)
    new_ob = np.vstack(ob)
    # print(new_ob.shape)
    new_ob = np.argmax(new_ob, axis=1)
    # print(new_ob.shape)
    action = net.activate(new_ob)
    action = process_action(action)
    return action



def eval_fitness(genome, config, candidate_params=None):
    fitness = []
    if candidate_params is None:
        candidate_params = runner.cmaes_model.get_current_parameters()
    runner.set_params(candidate_params)

    for _ in range(AttentionNEATConfig.TRIALS):
        net = RecurrentNetwork.create(genome, config)
        ob, info = env.reset()
        total_reward = 0
        step = 0
        done = False
        
        while not done:
            action = get_action(net, ob)
            ob, reward, done, trunc, info = env.step(action)
                
            if trunc:
                env.reset()
            step += 1
            total_reward += reward #+0.1
            #print(total_reward)
            #env.render()

        fitness.append(total_reward)

    return np.array(fitness).mean()


def test(genome):
    score_list, time_list = [], []
    for i in range(AttentionNEATConfig.TEST):
        start = time.time()
        score = eval_fitness(genome, AttentionNEATConfig.NEAT_CONFIG, None)
        end = time.time()

        print('\n#################### Test Result #####################\n')
        print('Reward: {0}'.format(score))
        print('Execution time: {0:.3f} sec'.format(end - start))
        print('\n######################################################\n')
        score_list.append(score)
        time_list.append(end - start)

    print('Mean Execution time: {0:.3f} sec'.format(np.array(time_list).mean()))
    print('Mean Test Fitness: {0:.3f}'.format(np.array(score_list).mean()))


def save_result(fitness: int, best_genome):
    net = RecurrentNetwork.create(best_genome, AttentionNEATConfig.NEAT_CONFIG)
    path = BASE_DIR + f'main_model_{fitness}.pkl'

    with open(path, 'wb') as net_output:
        pickle.dump(net, net_output, pickle.HIGHEST_PROTOCOL)

    with open(path, 'wb') as attention_neat_output:
        pickle.dump(runner, attention_neat_output, pickle.HIGHEST_PROTOCOL)


def load(fitness: int, reset=True):
    path = BASE_DIR + f'main_model_{fitness}.pkl'
    if reset or not os.path.isfile(path):
        return AttentionNEATModule()
    else:
        with open(path, 'rb') as attention_neat_output:
            return pickle.load(attention_neat_output)


def run(population, fitness: int, generations=AttentionNEATConfig.GENERATIONS):
    parallel_runner = ParallelEvaluator(CPU_COUNT,
                                        runner.cmaes_model,
                                        eval_fitness)

    winner = population.run(parallel_runner.evaluate,
                            generations,
                            parallel_runner.evaluate_cmaes_for_attention)
    save_result(winner)

if __name__ == '__main__':
    fitness = 0
    if len(sys.argv) > 1:
        fitness = int(sys.argv[1])
        gym.make('Snake-v1', render_mode="human", fitness=fitness)
    
    print(SelfAttentionConfig.IMAGE_SHAPE)
    runner = load(fitness, reset=False)
    run(runner.population, fitness)
