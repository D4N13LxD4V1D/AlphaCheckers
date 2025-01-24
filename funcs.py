import numpy as np
import random
from utils import setup_logger
from settings import run_folder, run_archive_folder
import loggers as lg

from game import Game, GameState
from model import Residual_CNN

from agent import Agent, User

import config
import copy

def playMatchesBetweenVersions(env, run_version, player1version, player2version, EPISODES, logger, turns_until_tau0, goes_first = 0, logger_stats=None, output_moves = False):

    if player1version == -1:
        player1 = User('player1', env.state_size, env.action_size)
    else:
        player1_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)

        if player1version > 0:
            player1_network = player1_NN.read(env.name, run_version, player1version)
            player1_NN.model.set_weights(player1_network.get_weights())   
        player1 = Agent(f'player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_NN)

    if player2version == -1:
        player2 = User('player2', env.state_size, env.action_size)
    else:
        player2_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
        
        if player2version > 0:
            player2_network = player2_NN.read(env.name, run_version, player2version)
            player2_NN.model.set_weights(player2_network.get_weights())
        player2 = Agent(f'player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player2_NN)

    scores, memory, points, sp_scores = playMatches(player1, player2, EPISODES, logger, turns_until_tau0, None, goes_first, logger_stats=logger_stats, run_version = run_version, output_moves=output_moves)

    return (scores, memory, points, sp_scores)


def playMatches(player1, player2, EPISODES, logger, turns_until_tau0, memory = None, goes_first = 0, logger_stats=None, run_version = None, output_moves=False):
    env = Game()
    p1, p2, draw = 0, 0, 0


    scores = {player1.name:0, "drawn": 0, player2.name:0}
    sp_scores = {'sp':0, "drawn": 0, 'nsp':0}
    points = {player1.name:[], player2.name:[]}

    for e in range(EPISODES):
        logger.info('====================')
        logger.info('EPISODE %d OF %d', e+1, EPISODES)
        logger.info('====================')
        print('====================')
        print(f'EPISODE {e+1} OF {EPISODES}')
        print('====================')

        print (str(e+1) + ' ', end='')

        state = env.reset()
        
        done = 0
        turn = 0
        player1.mcts = None
        player2.mcts = None

        if goes_first == 0:
            player1Starts = random.randint(0, 1) * 2 - 1
        else:
            player1Starts = goes_first

        if player1Starts == 1:
            players = {1:{"agent": player1, "name":player1.name}
                    , -1: {"agent": player2, "name":player2.name}
                    }
            logger.info(player1.name + ' plays as W')
        else:
            players = {1:{"agent": player2, "name":player2.name}
                    , -1: {"agent": player1, "name":player1.name}
                    }
            logger.info(player2.name + ' plays as W')
            logger.info('--------------')

        env.gameState.render(logger)
        while done == 0:
            turn = turn + 1

            logger.info("delta 1")
            #### Run the MCTS algo and return an action
            if turn < turns_until_tau0:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(copy.deepcopy(state), 1)
            else:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(copy.deepcopy(state), 0)
            logger.info("delta 2")

            if memory != None:
                ####Commit the move to memory
                memory.commit_stmemory(env.identities, state, pi)

            lg.logger_moves.info(action)
            logger.info('action: %d', int(action))
            logger.info(f"Current counter: {env.gameState.counter}")
            for r in range(env.grid_shape[0]):
                logger.info(['----' if x == 0 else '{0:.2f}'.format(np.round(x,2)) for x in pi[env.grid_shape[1]*r : (env.grid_shape[1]*r + env.grid_shape[1])]])

            # logger.info('MCTS perceived value for %s: %f', state.pieces[str(state.playerTurn)], np.round(MCTS_value, 2))
            # logger.info('NN perceived value for %s: %f', state.pieces[str(state.playerTurn)], np.round(NN_value, 2))
            logger.info('====================')

            ### Do the action
            state, value, done, _ = env.step(int(action)) #the value of the newState from the POV of the new playerTurn i.e. -1 if the previous player played a winning move

            env.gameState.render(logger)

            if done == 1:
                lg.logger_moves.info("====================")
                if memory != None:
                    #### If the game is finished, assign the values correctly to the game moves
                    for move in memory.stmemory:
                        if move['playerTurn'] == state.playerTurn:
                            move['value'] = value
                        else:
                            move['value'] = -value
                         
                    memory.commit_ltmemory()
             
                if value == 1:
                    p1 += 1
                    print('%s WINS!', players[state.playerTurn]['name'])
                    logger.info('%s WINS!', players[state.playerTurn]['name'])
                    scores[players[state.playerTurn]['name']] = scores[players[state.playerTurn]['name']] + 1
                    if state.playerTurn == 1: 
                        sp_scores['sp'] = sp_scores['sp'] + 1
                    else:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1

                elif value == -1:
                    p2 += 1
                    print('%s WINS!', players[-state.playerTurn]['name'])
                    logger.info('%s WINS!', players[-state.playerTurn]['name'])
                    scores[players[-state.playerTurn]['name']] = scores[players[-state.playerTurn]['name']] + 1
               
                    if state.playerTurn == 1: 
                        sp_scores['nsp'] = sp_scores['nsp'] + 1
                    else:
                        sp_scores['sp'] = sp_scores['sp'] + 1

                else:
                    draw += 1
                    logger.info('DRAW...')
                    print("DRAW...")
                    scores['drawn'] = scores['drawn'] + 1
                    sp_scores['drawn'] = sp_scores['drawn'] + 1

                pts = state.score
                points[players[state.playerTurn]['name']].append(pts[0])
                points[players[-state.playerTurn]['name']].append(pts[1])

    return (scores, memory, points, sp_scores)
