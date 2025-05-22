""" Gymnasium implementation of the Snake game by Jakub Łyskawa.
Original implementation for gym by Ming Yu: https://github.com/NaLooo/Gym_Snake_Game
"""
from gymnasium_snake_game.version import VERSION as __version__
from gymnasium_snake_game.environment import SnakeEnv

from gymnasium.envs.registration import register


import gymnasium

__author__ = "Jakub Łyskawa, Ming Yu"

register(
    id="Snake-v1",
    entry_point="gymnasium_snake_game.environment:SnakeEnv",
)


def make(name, **kwargs):
    return gym.make(name, **kwargs)
