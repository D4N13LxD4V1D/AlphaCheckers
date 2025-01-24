import numpy as np
import loggers as lg
import random

class Game:

	def __init__(self):
		self.currentPlayer = -1  # 1 or -1 for two different players
		self.gameState = GameState(self.initBoard(), playerTurn=-1)
		self.actionSpace = np.full((64,), fill_value=0, dtype=np.int)
		self.pieces = {'2': 'R', '1': 'r', '0': '-', '-1': 'b',
					   '-2': 'B'}  # R-red queen, r-red token; B-black queen, b-black token
		self.grid_shape = (8, 8)
		self.input_shape = (2, 8, 8)
		self.name = 'checkers'
		self.state_size = len(self.gameState.binary)
		self.action_size = len(self.actionSpace)

	def initBoard(self):
		r"""
		Notes:
			Creates new blank board and fills white and black pieces on it (on dark squares)
		Returns:
			numpy array size (64,) with starting board representation

		"""
		board = np.full((64,), fill_value=0, dtype=np.int)

		#black pieces
		board[1:8:2] = 1
		board[8:15:2] = 1
		board[17:24:2] = 1

		# white pieces
		board[40:47:2] = -1
		board[49:56:2] = -1
		board[56:63:2] = -1


		return board

	def reset(self):
		r"""
		Notes:
			Sets game state to blank state(0), and current player to 1
		Returns:
			New fresh state

		"""
		self.gameState = GameState(self.initBoard(), playerTurn=-1)
		self.currentPlayer = -1
		return self.gameState

	def step(self, action):
		r"""
		Notes:
			switches current player and updates current game state
		Args:
			action: number representing action on board to take

		Returns:
			(tuple):
				next_state: (new board state),
				value: (1 previous player has won, 0 otherwise)
				done: (1 if game is finished, 0 otherwise), info(None)


		"""
		next_state, value, done = self.gameState.takeAction(action)
		self.gameState = next_state
		self.currentPlayer = -self.currentPlayer
		info = None
		return next_state, value, done, info

	def identities(self, state, actionValues):
		"""
		Notes:
			Flipps board on existing symetries.
		See Also:
			funcs.py -> memory.commit_stmemory(env.identities, state, pi)
			memory.py -> commit_stmemory()
		Args:
			state:
			actionValues:

		Returns:

		"""

		identities = [(state, actionValues)]

		currentBoard = state.board
		currentAV = actionValues

		currentBoard = np.array([
			currentBoard[7], currentBoard[6], currentBoard[5], currentBoard[4], currentBoard[3], currentBoard[2], currentBoard[1], currentBoard[0]
			, currentBoard[15], currentBoard[14], currentBoard[13], currentBoard[12], currentBoard[11], currentBoard[10], currentBoard[9], currentBoard[8]
			, currentBoard[23], currentBoard[22], currentBoard[21], currentBoard[20], currentBoard[19], currentBoard[18], currentBoard[17], currentBoard[16]
			, currentBoard[31], currentBoard[30], currentBoard[29], currentBoard[28], currentBoard[27], currentBoard[26], currentBoard[25], currentBoard[24]
			, currentBoard[39], currentBoard[38], currentBoard[37], currentBoard[36], currentBoard[35], currentBoard[34], currentBoard[33], currentBoard[32]
			, currentBoard[47], currentBoard[46], currentBoard[45], currentBoard[44], currentBoard[43], currentBoard[42], currentBoard[41], currentBoard[40]
			, currentBoard[55], currentBoard[54], currentBoard[53], currentBoard[52], currentBoard[51], currentBoard[50], currentBoard[49], currentBoard[48]
			, currentBoard[63], currentBoard[62], currentBoard[61], currentBoard[60], currentBoard[59], currentBoard[58], currentBoard[57], currentBoard[56]
		])

		currentAV = np.array([
			currentAV[7], currentAV[6], currentAV[5], currentAV[4], currentAV[3], currentAV[2], currentAV[1], currentAV[0]
			, currentAV[15], currentAV[14], currentAV[13], currentAV[12], currentAV[11], currentAV[10], currentAV[9], currentAV[8]
			, currentAV[23], currentAV[22], currentAV[21], currentAV[20], currentAV[19], currentAV[18], currentAV[17], currentAV[16]
			, currentAV[31], currentAV[30], currentAV[29], currentAV[28], currentAV[27],currentAV[26], currentAV[25], currentAV[24]
			, currentAV[39], currentAV[38], currentAV[37], currentAV[36], currentAV[35], currentAV[34], currentAV[33], currentAV[32]
			, currentAV[47], currentAV[46], currentAV[45], currentAV[44], currentAV[43], currentAV[42], currentAV[41], currentAV[40]
			, currentAV[55], currentAV[54], currentAV[53], currentAV[52], currentAV[51], currentAV[50], currentAV[49], currentAV[48]
			, currentAV[63], currentAV[62], currentAV[61], currentAV[60], currentAV[59], currentAV[58], currentAV[57], currentAV[56]
		])

		identities.append((GameState(currentBoard, state.playerTurn), currentAV))

		return identities


class GameState:

	def __init__(self, board, playerTurn, counter=0):
		self.board = board
		self.counter = counter
		self.pieces = {'2': 'R', '1': 'r', '0': '-', '-1': 'b',
					   '-2': 'B'}  # R-red queen, r-red token; B-black queen, b-black token

		self.playerTurn = playerTurn
		self.binary = self._binary()
		self.id = self._convertStateToId()
		self.allowedActions = self._allowedActions()
		self.isEndGame = self._checkForEndGame()
		self.value = self._getValue()
		self.score = self._getScore()

	def _allowedActions(self):
		r"""
		Returns:
			List: list of allowed states
		"""
		allowed = []

		tmp = self._check_all()

		for x in tmp:
			allowed.append(x[1])

		random.shuffle(allowed)

		return allowed

	def _binary(self):
		r"""
		Notes:
			creates array of size 2xboardSize, first half represents first player positions,
			second half representes second player positions
		Returns:
			Array of size 2xBoardSize

		"""
		currentplayer_position = np.zeros(len(self.board), dtype=np.int)
		currentplayer_position[self.board == self.playerTurn] = 1
		currentplayer_position[self.board == 2*self.playerTurn] = 2

		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board == -self.playerTurn] = 1
		other_position[self.board == 2*-self.playerTurn] = 2

		position = np.append(currentplayer_position, other_position)

		return position

	def _convertStateToId(self):
		r"""
			Notes:
				creates array of size 2xboardSize, first half represents first player positions,
				second half representes second player positions. Then converts this to string
			Returns:
				String of length 2xBoardSize

			"""

		player1_position = np.zeros(len(self.board), dtype=np.int)
		player1_position[self.board == 1] = 1
		player1_position[self.board == 2] = 2

		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board == -1] = -1
		other_position[self.board == -2] = -2

		stevec = np.array([self.counter], dtype=np.int)

		position = np.append(player1_position, other_position)
		position = np.append(position, stevec)

		id = ''.join(map(str, position))

		return id

	def _checkForEndGame(self):

		r"""
		Notes:
			Checks if current board is finished/won.

		Returns:
			int: 1(if game is finished/won), 0(otherwise)

		"""
		# checking if any player has lost all pieces
		unique, counts = np.unique(self.board, return_counts=True)
		count = dict(zip(unique, counts))

		black = count.get(-2, 0) + count.get(-1, 0)
		white = count.get(2, 0) + count.get(1, 0)

		if black == 0 or white == 0:
			return 1

		if self.counter >= 40:
			return 1

		#if player has no possible moves left
		if self._allowedActions() == []:
			return 1

		return 0

	def _getValue(self):
		r"""
		Notes:
			second and third value represents first an second player
		Returns:
			Tuple: 0,0,0 or -1,-1,1

		"""
		# This is the value of the state for the current player
		# i.e. if the previous player played a winning move, you lose

		unique, counts = np.unique(self.board, return_counts=True)
		count = dict(zip(unique, counts))

		if 2*self.playerTurn in count and self.playerTurn in count:
			current = count[2*self.playerTurn] + count[self.playerTurn]
		elif self.playerTurn in count:
			current = count[self.playerTurn]
		else:
			current = 0

		if self.counter >= 40:
			return 0, 0, 0

		if current == 0:
			return -1, -1, 1

		if self._allowedActions() == [] or self._allowedActions() == None:
			return -1, -1, 1

		return 0, 0, 0

	def _getScore(self):
		r"""
		See Also:
			_getValue
		Returns:

		"""
		tmp = self.value
		return tmp[1], tmp[2]

	def check_RU(self, x, jumps):  # checks / UP  (-7) diagonal
		tmp = []
		old = jumps
		if x % 8 < 6 and x > 15:  # right limit and upper limit
			if self.board[x - 7] in {-self.playerTurn, 2 * -self.playerTurn} and self.board[
				x - 2 * 7] == 0:  # jump over opponent
				if jumps == 1:
					tmp.append((x, x - 2 * 7))
				else:
					tmp.clear()
					tmp.append((x, x - 2 * 7))
					jumps = 1
		if x % 8 < 7 and x > 7:  # right limit and upper limit
			if self.board[x - 7] == 0 and jumps != 1:
				tmp.append((x, x - 7))
		return tmp, not (jumps == old)

	def check_LU(self, x, jumps):  # checks \ UP  (-9) diagonal
		tmp = []
		old = jumps
		if x % 8 > 1 and x > 15:  # left limit and upper limit
			if self.board[x - 9] in {-self.playerTurn, 2 * -self.playerTurn} and self.board[
				x - 2 * 9] == 0:  # jump over opponent
				if jumps == 1:
					tmp.append((x, x - 2 * 9))
				else:
					tmp.clear()
					tmp.append((x, x - 2 * 9))
					jumps = 1
		if x % 8 > 0 and x > 7:  # left limit and upper limit
			if self.board[x - 9] == 0 and jumps != 1:
				tmp.append((x, x - 9))
		return tmp, not (jumps == old)

	def check_LD(self, x, jumps):  # checks \ DOWN  (+9) diagonal
		tmp = []
		old = jumps
		if x % 8 < 6 and x < 48:  # right limit and lower limit
			if self.board[x + 9] in {-self.playerTurn, 2 * -self.playerTurn} and self.board[
				x + 2 * 9] == 0:  # jump over opponent
				if jumps == 1:
					tmp.append((x, x + 2 * 9))
				else:
					tmp.clear()
					tmp.append((x, x + 2 * 9))
					jumps = 1
		if x % 8 < 7 and x < 56:  # right limit and lower limit
			if self.board[x + 9] == 0 and jumps != 1:
				tmp.append((x, x + 9))
		return tmp, not (jumps == old)

	def check_RD(self, x, jumps):  # checks / DOWN  (+7) diagonal
		tmp = []
		old = jumps

		if x % 8 > 1 and x < 48:  # left limit and lower limit
			if self.board[x + 7] in {-self.playerTurn, 2 * -self.playerTurn} and self.board[
				x + 2 * 7] == 0:  # jump over opponent
				if jumps == 1:
					tmp.append((x, x + 2 * 7))
				else:
					tmp.clear()
					tmp.append((x, x + 2 * 7))
					jumps = 1
		if x % 8 > 0 and x < 56:  # left limit and lower limit
			if self.board[x + 7] == 0 and jumps != 1:
				tmp.append((x, x + 7))

		return tmp, not (jumps == old)

	def _check_all(self):
		tmp = []
		jumps = 0

		cur = np.where(self.board == self.playerTurn)[0]  # regular pieces
		cur1 = np.where(self.board == 2 * self.playerTurn)[0]  # king pieces

		if self.playerTurn == -1:
			for x in cur:  # normal piece
				a, reset = self.check_LU(x, jumps)
				if reset:
					tmp.clear()
					jumps = 1
					tmp.extend(a)
				else:
					tmp.extend(a)
				b, reset = self.check_RU(x, jumps)
				if reset:
					tmp.clear()
					jumps = 1
					tmp.extend(b)
				else:
					tmp.extend(b)

			for x in cur1:  # king piece
				a, reset = self.check_LU(x, jumps)
				if reset:
					tmp.clear()
					jumps = 1
					tmp.extend(a)
				else:
					tmp.extend(a)
				b, reset = self.check_RU(x, jumps)
				if reset:
					tmp.clear()
					jumps = 1
					tmp.extend(b)
				else:
					tmp.extend(b)

				c, reset = self.check_LD(x, jumps)
				if reset:
					tmp.clear()
					jumps = 1
					tmp.extend(c)
				else:
					tmp.extend(c)
				d, reset = self.check_RD(x, jumps)
				if reset:
					tmp.clear()
					jumps = 1
					tmp.extend(d)
				else:
					tmp.extend(d)

		if self.playerTurn == 1:
			for x in cur:  # normal pieces
				# check / up diagonal
				a, reset = self.check_RD(x, jumps)
				if reset:
					tmp.clear()
					jumps = 1
					tmp.extend(a)
				else:
					tmp.extend(a)
				b, reset = self.check_LD(x, jumps)
				if reset:
					tmp.clear()
					jumps = 1
					tmp.extend(b)
				else:
					tmp.extend(b)

			for x in cur1:  # king piece
				a, reset = self.check_LU(x, jumps)
				if reset:
					tmp.clear()
					jumps = 1
					tmp.extend(a)
				else:
					tmp.extend(a)
				b, reset = self.check_RU(x, jumps)
				if reset:
					tmp.clear()
					jumps = 1
					tmp.extend(b)
				else:
					tmp.extend(b)

				c, reset = self.check_LD(x, jumps)
				if reset:
					tmp.clear()
					jumps = 1
					tmp.extend(c)
				else:
					tmp.extend(c)
				d, reset = self.check_RD(x, jumps)
				if reset:
					tmp.clear()
					jumps = 1
					tmp.extend(d)
				else:
					tmp.extend(d)
		return tmp

	def takeAction(self, action):
		"""
		Args:
			action: number representing position on board

		Returns:
			Tuple: newState-(Board representing new state), value-(1 current player has won, 0 otherwise) , done(1 if game is finished, 0 otherwise)

		"""

		if action >= 63 or action < 0:
			raise IndexError("Action is not within allowed values")

		possible = self._check_all()
		move = None

		for x in possible:
			if x[1] == action:
				move = x

		if move is None:
			lg.logger_main.info(f"Error occured: wanted action: {action} possible actions :{possible}")
			self.render(lg.logger_main)
			print(action)
			print(possible)
			raise IndexError("Action is not in the list of possible actions")

		delta = move[1] - move[0]
		if abs(delta) > 9:  # move was a jump over opponent piece
			newBoard = np.array(self.board)
			newBoard[move[1]] = newBoard[move[0]]
			newBoard[int(move[0] + (delta / 2))] = 0
			newBoard[move[0]] = 0
			self.counter = 0
		else:
			newBoard = np.array(self.board)
			newBoard[move[1]] = newBoard[move[0]]
			newBoard[move[0]] = 0
			self.counter += 1
		# if pieces is in last/first row and it's not allready king piece
		if (0 <= move[1] <= 7 or 56 <= move[1] <= 63) and abs(newBoard[move[1]]) == 1:
			newBoard[move[1]] *= 2  # change piece to king piece of the same color

		newState = GameState(newBoard, -self.playerTurn, self.counter)

		value = 0
		done = 0

		if newState.isEndGame:
			value = newState.value[0]
			done = 1

		# pieces = {'2': 'R', '1': 'r', '0': ' ', '-1': 'b', '-2': 'B'}
		# for r in range(8):
		# 	lg.logger_board.info([pieces[str(x)] for x in newBoard[8 * r: (8 * r + 8)]])
		# lg.logger_board.info(f"Counter {self.counter}")
		# lg.logger_board.info('--------------')

		return newState, value, done

	def render(self, logger):
		r"""
		Notes:
			Renders a game state in corresponding logger (switches numbers with character representation)
		Args:
			logger: which logger to write to

		Returns:
			None

		"""
		for r in range(8):
			logger.info([self.pieces[str(x)] for x in self.board[8 * r: (8 * r + 8)]])
		logger.info('--------------')
