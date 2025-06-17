import pickle
import time
import gymnasium
import numpy as np
import imageio
import pygame 

# Assuming these are available in your environment from the original project
from experiment.runner import AttentionNEATModule
from experiment.configs.config import BASE_DIR, AttentionNEATConfig
from neat.nn.recurrent import RecurrentNetwork
from utility import process_action

# Import the SnakeEnv and Snake classes
from experiment.configs.config import *


def get_action(net, ob):
    """
    Activates the neural network to get an action from the observation.
    """
    # The 'ob' here is already the processed input from Snake.get_inputs()
    action = net.activate(ob)
    action = process_action(action)
    return action


def load_and_play(model_path=BASE_DIR + 'main_model.pkl', fps=5):
    """
    Loads the best genome from a NEAT model and makes it play the Snake game.
    """
    try:
        with open(model_path, 'rb') as f:
            runner = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please ensure `main_model.pkl` exists.")
        return

    # Get the best genome from the loaded runner
    best_genome = runner.population.best_genome
    if best_genome is None:
        print("Error: No best genome found in the loaded model. The model might not have been trained.")
        return

    # Create the neural network from the best genome
    net = RecurrentNetwork.create(best_genome, AttentionNEATConfig.NEAT_CONFIG)

    print(f"\n--- Starting playback of the trained model at {fps} FPS ---")
    print("Press 'q' or close the window to quit.")

    # Calculate the time per frame
    if fps > 0:
        time_per_frame = 1.0 / fps
    else:
        time_per_frame = 0 # No delay if fps is 0 or less

    try:
        ob, info = env.reset()
        total_reward = 0
        done = False
        truncated = False

        initial_surface = env.render() 
        height, width = initial_surface.get_height(), initial_surface.get_width()
        
        # Initialize video writer
        video_writer = imageio.get_writer("snake_video.mp4", fps=fps, codec='libx264', quality=8, macro_block_size=1)
        
        # Add the first frame
        frame_array = pygame.surfarray.array3d(initial_surface)
        frame_array = np.transpose(frame_array, (1, 0, 2)) # Transpose for (H, W, C)
        video_writer.append_data(frame_array)


        while not done and not truncated:
            start_time = time.time() # Record start time of the frame

            # The environment's step function expects an action to be passed
            # The get_action function takes the neural network and observation
            action = get_action(net, ob)

            # Pass the action to the environment
            ob, reward, done, truncated, info = env.step(action)
            
            current_surface = env.render()
            frame_array = pygame.surfarray.array3d(current_surface)
            frame_array = np.transpose(frame_array, (1, 0, 2)) # Transpose for (H, W, C)
            video_writer.append_data(frame_array)
            
            total_reward += reward

            # If the game is done (e.g., snake crashed), reset for continuous play
            if done or truncated:
                print(truncated, info)
                print(f"Game Over! Total Reward: {total_reward:.2f}")
                print("Resetting game...")
                ob, info = env.reset()
                total_reward = 0

            # Enforce FPS
            end_time = time.time()
            elapsed_time = end_time - start_time
            if time_per_frame > elapsed_time:
                time.sleep(time_per_frame - elapsed_time)
    
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    finally:
        video_writer.close()
        env.close()
        print("Environment closed.")


if __name__ == '__main__':
    env = gym.make('Snake-v1', render_mode="human")
    # You can change the FPS here to make it faster or slower
    load_and_play(fps=30) # Play at 5 frames per second
    # load_and_play(fps=15) # Play at 15 frames per second
    # load_and_play(fps=0) # Play as fast as possible (no delay)