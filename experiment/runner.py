import os
import pickle
import time
import sys 
import csv 

from base_runner import BaseAttentionRunnerModule
from cma_es import CMAEvolutionStrategy
from experiment.configs.config import *
from neat.nn.recurrent import RecurrentNetwork
from parallel import ParallelEvaluator
from self_attention import SelfAttention
from utility import process_action, initial_population

fitness = 0

class AttentionNEATModule(BaseAttentionRunnerModule):
    def __init__(self, fitness):
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
        self.population = initial_population(None, __stats, AttentionNEATConfig.NEAT_CONFIG, True, fitness)


def get_action(net, ob):
    top = runner.attention_model.get_output(ob)  # patch centers (coords)
    new_ob_coords = runner.attention_model.normalize_patch_centers(top)

    # Extract actual pixel data for each top patch
    patch_pixels = []
    for x,y in top.detach().cpu().numpy().astype(int):
        patch = int(np.argmax(ob, axis=-1)[x,y])
        patch_pixels.append(patch)

    combined_input = np.concatenate([new_ob_coords, patch_pixels, [1.0]])

    action = net.activate(combined_input)
    action = process_action(action)
    return action, top

def eval_fitness(genome, config, seed, candidate_params=None):
    global fitness
    
    if not isinstance(seed, int):
        seed = 0
    fitness_values = []

    apples_picked_up = []

    if candidate_params is None:
        candidate_params = runner.cmaes_model.get_current_parameters()
    runner.set_params(candidate_params)
    env.unwrapped.set_seed(seed)

    # Create the dynamic CSV filename based on fitness
    csv_filename = f'attention_{fitness}_{seed}.csv'
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        for _ in range(AttentionNEATConfig.TRIALS):
            net = RecurrentNetwork.create(genome, config)
            ob, info = env.reset()
            env.unwrapped.set_seed(seed)
            total_reward = 0
            total_apples = 0
            step = 0
            done = False

            while not done:
                action, top = get_action(net, ob)
                # Write attention output (top) to the CSV
                if hasattr(top, 'detach'):
                    top_array = top.detach().cpu().numpy()
                else:
                    top_array = np.array(top)

                # Convert each row (x, y) into a tuple string like "(x, y)"
                top_values = [f"({x:.1f}, {y:.1f})" for x, y in top_array]

                # Write the list of tuple strings as a single row
                writer.writerow(top_values)

                ob, (reward, apples), done, trunc, info = env.step(action)

                if done:
                    reward -= 5
                if trunc:
                    env.reset()
                    env.unwrapped.set_seed(seed)
                step += 1
                total_reward += reward
                total_apples += apples

            fitness_values.append(total_reward)
            apples_picked_up.append(total_apples)

    return np.array(fitness_values).mean(), np.array(apples_picked_up).mean()


def test(genome):
    score_list, time_list = [], []
    for i in range(AttentionNEATConfig.TEST):
        start = time.time()
        score, apples = eval_fitness(genome, AttentionNEATConfig.NEAT_CONFIG, 0, None)
        end = time.time()

        print('\n#################### Test Result #####################\n')
        print('Reward: {0}'.format(score))
        print('Execution time: {0:.3f} sec'.format(end - start))
        print('\n######################################################\n')
        score_list.append(score)
        time_list.append(end - start)

    print('Mean Execution time: {0:.3f} sec'.format(np.array(time_list).mean()))
    print('Mean Test Fitness: {0:.3f}'.format(np.array(score_list).mean()))


def save_result(best_genome, fitness):
    net = RecurrentNetwork.create(best_genome, AttentionNEATConfig.NEAT_CONFIG)

    with open(BASE_DIR + f'net_output_{fitness}.pkl', 'wb') as net_output:
        pickle.dump(net, net_output, pickle.HIGHEST_PROTOCOL)

    with open(BASE_DIR + f'main_model_{fitness}.pkl', 'wb') as attention_neat_output:
        pickle.dump(runner, attention_neat_output, pickle.HIGHEST_PROTOCOL)


def load(fitness, reset=True):
    if reset or not os.path.isfile(BASE_DIR + f'main_model_{fitness}.pkl'):
        return AttentionNEATModule(fitness)
    else:
        with open(BASE_DIR + f'main_model_{fitness}.pkl', 'rb') as attention_neat_output:
            return pickle.load(attention_neat_output)


def run(population, fitness, generations=AttentionNEATConfig.GENERATIONS):
    parallel_runner = ParallelEvaluator(CPU_COUNT,
                                        runner.cmaes_model,
                                        eval_fitness)
    seed_value=0
    
    def update_val():
        nonlocal seed_value
        seed_value+=1
        
    def eval(a,b, *args):
        nonlocal seed_value
        parallel_runner.evaluate(a,b,seed_value, *args)

    def evalcmaes(a,b, *args):
        nonlocal seed_value
        parallel_runner.evaluate_cmaes_for_attention(a,b,seed_value, *args)

    winner = population.run(update_val,
                            lambda a,b, *args: eval(a,b, *args),
                            generations,
                            lambda a,b, *args: evalcmaes(a,b, *args))
    
    save_result(winner, fitness)

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        fitness = int(sys.argv[1])  # Take fitness from command-line argument
        # env = gym.make('Snake-v1', render_mode="human", fitness=fitness)  # Set fitness in environment creation
        env = gym.make('Snake-v1', render_mode=None, fitness=fitness)
    print(SelfAttentionConfig.IMAGE_SHAPE)
    runner = load(fitness, reset=False)
    run(runner.population, fitness)
