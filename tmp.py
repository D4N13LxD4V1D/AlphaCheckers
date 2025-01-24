from keras.utils import plot_model
import numpy as np
np.set_printoptions(suppress=True)

from shutil import copyfile
import random
from importlib import reload


from keras.utils import plot_model

from game import Game, GameState
from agent import Agent
from memory import Memory
from model import Residual_CNN
from funcs import playMatches, playMatchesBetweenVersions

import loggers as lg

from settings import run_folder, run_archive_folder
import initialise
import pickle
import config


env = Game()
current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + env.grid_shape, env.action_size,
						  config.HIDDEN_CNN_LAYERS)


plot_model(current_NN.model, to_file=run_folder + 'models/model.png', show_shapes = True)