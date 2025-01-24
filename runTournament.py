import numpy as np
np.set_printoptions(suppress=True)
from game import Game
from funcs import playMatchesBetweenVersions
import loggers as lg

# for v1 in range (0, 10):
v1 = 10
for v2 in range(8, 9):
	env = Game()
	scores, memory, points, sp_scores = playMatchesBetweenVersions(env, run_version=1, player1version=v1, player2version=v2, EPISODES=50, logger=lg.logger_tourney, turns_until_tau0=0)
	p1 = scores['player1']
	dr = scores['drawn']
	p2 = scores['player2']
	scoreP1 = int(p1) * 3 + dr
	scoreP2 = int(p2) * 3 + dr

	print(f"v{v1}: {scores['player1']}\t draw: {scores['drawn']}\t v{v2}: {scores['player2']}")
	lg.logger_tournament.info(f"v{v1}: {scores['player1']}\t draw: {scores['drawn']}\t v{v2}: {scores['player2']}")
	lg.logger_tournament.info(f"v{v1}:{scoreP1}\t v{v2}:{scoreP2}")

# v1 = 9
# v2 = 3
# env = Game()
# scores, memory, points, sp_scores = playMatchesBetweenVersions(env, run_version=1, player1version=v1,
# 																	   player2version=v2, EPISODES=50,
# 																	   logger=lg.logger_tourney, turns_until_tau0=0)
# print(f"v{v1}: {scores['player1']} \t v{v2}: {scores['player2']} \t draw: {scores['drawn']}")
# lg.logger_tournament.info(f"v{v1}: {scores['player1']} \t v{v2}: {scores['player2']} \t draw: {scores['drawn']}")

