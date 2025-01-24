
from game import Game
from funcs import playMatchesBetweenVersions
import loggers as lg
import sys
from utils import setup_logger


RUN_VER = 2

env = Game()
logger_stat = setup_logger('logger_stat', "./run_archive/" + env.name + "/run" + str(RUN_VER).zfill(4) + '/logs/logger_stats.log')

for x in range(1, 10):
    scores, memory, points, sp_scores = playMatchesBetweenVersions(env=env, run_version = RUN_VER, player1version = x, player2version = x+1, EPISODES = 20, logger = lg.logger_tourney, goes_first=0, turns_until_tau0 = 0, logger_stats=lg.logger_stats)
    print(f"Score: {scores}")
    print(f"SP Score: {sp_scores}")
    logger_stat.info(f"P1 ver.: {x} \t P2 ver.: {x+1}")
    logger_stat.info(f"Score: {scores}")
    logger_stat.info(f"SP Score: {sp_scores}")

#
#
# scores, memory, points, sp_scores = playMatchesBetweenVersions(env=env, run_version = 1, player1version = int(sys.argv[2]), player2version = int(sys.argv[1]), EPISODES = int(sys.argv[3]), logger = lg.logger_tourney, goes_first=0, turns_until_tau0 = 0, logger_stats=lg.logger_stats)
# print(f"Score: {scores}")
# print(f"SP Score: {sp_scores}")
# logger_stat.info(f"P1 ver.: {int(sys.argv[2])} \t P2 ver.: {int(sys.argv[1])}")
# logger_stat.info(f"Score: {scores}")
# logger_stat.info(f"SP Score: {sp_scores}\n\n")
