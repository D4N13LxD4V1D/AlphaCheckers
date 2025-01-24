
from game import Game
from funcs import playMatchesBetweenVersions
import loggers as lg
import sys
from utils import setup_logger
import numpy as np
import threading
from multiprocessing.pool import ThreadPool
import time


RUN_VER = 2

env = Game()
logger_stat = setup_logger('logger_stat', "./run_archive/" + env.name + "/run" + str(RUN_VER).zfill(4) + '/logs/logger_stats.log')



#
# a = np.ndarray(shape=(10, 10), dtype=int)
#
# for x in range(1,10):
#     for y in range(1,10):
#         print(f"\nVer.{x} VS Ver.{y}")
#         scores, memory, points, sp_scores = playMatchesBetweenVersions(env=env, run_version=RUN_VER,
#                                                                        player1version=x, player2version=y,
#                                                                        EPISODES=10, logger=lg.logger_tourney,
#                                                                        goes_first=0, turns_until_tau0=0,
#                                                                        logger_stats=lg.logger_stats)
#         score = scores["player1"] - scores["player2"]
#         print(f"SCORE: {score}")
#         print(f"Score: {scores}")
#         print(f"SP Score: {sp_scores}")
#         logger_stat.info(f"P1 ver.: {x} \t P2 ver.: {x+1}")
#         logger_stat.info(f"Score: {scores}")
#         logger_stat.info(f"SP Score: {sp_scores}")
#         a[x, y] = score
#
# np.savetxt("./run_archive/" + env.name + "/run" + str(RUN_VER).zfill(4) + '/logs/tournament_data.csv', a.data, delimiter=",", fmt="%d")

#env=env,     run_version=RUN_VER,   player1version=x,    player2version=y,    EPISODES=10,     logger=lg.logger_tourney,  goes_first=0,    turns_until_tau0=0,      logger_stats=lg.logger_stats

def Test(a,b):
    print(f"Thread {a} | {b}")
    time.sleep(b/2)
    return a+1, b+1


pool = ThreadPool(processes=3)

a = []
for x in range(2):
    a.append((env,RUN_VER,1,x,1,lg.logger_tourney,0,0,lg.logger_stats))


res = pool.starmap(playMatchesBetweenVersions, a)
print(res)
