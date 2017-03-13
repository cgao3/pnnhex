from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from utils.read_data import BOARD_SIZE, INPUT_DEPTH, INPUT_WIDTH
from zobrist.zobrist import HexColor

import numpy as np

NORTH_EDGE=-1
SOUTH_EDGE=-3

EAST_EDGE=-2
WEST_EDGE=-4

class MoveConvertUtil:
    def __init__(self):
        pass

    @staticmethod
    def intMoveToPair(intMove):
        x=intMove//BOARD_SIZE
        y=intMove%BOARD_SIZE
        return (x,y)

    @staticmethod
    def intPairToIntMove(pair):
        x,y=pair
        return x*BOARD_SIZE+y

    @staticmethod
    def rawMoveToIntMove(rawMove):
        x = ord(rawMove[0].lower()) - ord('a')
        y = int(rawMove[1:]) - 1
        return x * BOARD_SIZE + y

    @staticmethod
    def intMoveToRaw(intMove):
        x, y = MoveConvertUtil.intMoveToPair(intMove)
        y += 1
        return chr(x + ord('a')) + repr(y)

def state_to_str(g):
    size=BOARD_SIZE
    white = 'W'
    black = 'B'
    empty = '.'
    ret = '\n'
    coord_size = len(str(size))
    offset = 1
    ret+=' '*(offset+1)
    board=[None]*size
    for i in range(size):
        board[i]=[empty]*size

    for k, i in enumerate(g):
        x,y=i//size, i%size
        board[x][y]=black if k%2==0 else white

    PLAYERS = {"white": white, "black": black}
    for x in range(size):
        ret += chr(ord('A') + x) + ' ' * offset * 2
    ret += '\n'
    for y in range(size):
        ret += str(y + 1) + ' ' * (offset * 2 + coord_size - len(str(y + 1)))
        for x in range(size):
            if (board[x][y] == PLAYERS["white"]):
                ret += white
            elif (board[x][y] == PLAYERS["black"]):
                ret += black
            else:
                ret += empty
            ret += ' ' * offset * 2
        ret += white + "\n" + ' ' * offset * (y + 1)
    ret += ' ' * (offset * 2 + 1) + (black + ' ' * offset * 2) * size

    return ret

def next_player(current_player):
    return (current_player + 1) % 2

class GameCheckUtil:
    def __init__(self):
        pass

    @staticmethod
    def winner(black_group, white_group):
        if (black_group.connected(NORTH_EDGE, SOUTH_EDGE)):
            return HexColor.BLACK
        elif (white_group.connected(WEST_EDGE, EAST_EDGE)):
            return HexColor.WHITE
        else:
            return HexColor.EMPTY

    @staticmethod
    def updateUF(intgamestate, black_group, white_group, intmove, player):
        assert(player == HexColor.BLACK or player== HexColor.WHITE)
        x, y = intmove // BOARD_SIZE, intmove % BOARD_SIZE
        neighbors = []
        pattern = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, 1), (1, -1)]
        for p in pattern:
            x1, y1 = p[0] + x, p[1] + y
            if 0 <= x1 < BOARD_SIZE and 0 <= y1 < BOARD_SIZE:
                neighbors.append((x1, y1))
        if (player == HexColor.BLACK):
            if (y == 0):
                black_group.join(intmove, NORTH_EDGE)
            if (y == BOARD_SIZE - 1):
                black_group.join(intmove, SOUTH_EDGE)

            for m in neighbors:
                m2 = m[0] * BOARD_SIZE + m[1]
                if (m2 in intgamestate and list(intgamestate).index(m2) % 2 == player-1):
                    black_group.join(m2, intmove)
        else:

            if (x == 0):
                white_group.join(intmove, WEST_EDGE)
            if (x == BOARD_SIZE - 1):
                white_group.join(intmove, EAST_EDGE)

            for m in neighbors:
                im = m[0] * BOARD_SIZE + m[1]
                if (im in intgamestate and list(intgamestate).index(im) % 2 == player-1):
                    white_group.join(im, intmove)
        # print(black_group.parent)
        return (black_group, white_group)

# input is raw score such as [-20,30,10]
def softmax_selection(logits, currentstate, temperature=1.0):
    logits = np.squeeze(logits)
    empty_positions = [i for i in range(BOARD_SIZE ** 2) if i not in currentstate]
    # print("empty positions:", empty_positions)
    # print(logits)
    effective_logits = [logits[i] for i in empty_positions]
    max_value = np.max(effective_logits)
    effective_logits = effective_logits - max_value
    effective_logits = np.exp(effective_logits)/temperature
    sum_value = np.sum(effective_logits)
    prob = effective_logits / sum_value
    #print(prob)
    v = np.random.random()
    sum_v = 0.0
    action = None
    for i, e in enumerate(prob):
        sum_v += e
        if (sum_v >= v):
            action = i
            break
    ret = empty_positions[action]
    del empty_positions, effective_logits
    return ret

# input is raw score such as [-20,30,10]
def max_selection(logits, currentstate):
    logits = np.squeeze(logits)
    empty_positions = [i for i in range(BOARD_SIZE ** 2) if i not in currentstate]
    # print("empty positions:", empty_positions)
    # print(logits)
    effective_logits = [logits[i] for i in empty_positions]
    max_ind=np.argmax(effective_logits)
    ret = empty_positions[max_ind]
    del empty_positions, effective_logits
    return ret

class RLTensorUtil:
    def __init__(self):
        pass

    @staticmethod
    def makeTensorInBatch(batchPositionTensors, kth, gamestate):
        #gamestate is an integer array of moves

        batchPositionTensors[kth, 1:INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, INPUT_DEPTH - 1] = 1
        # black occupied
        batchPositionTensors[kth, 0:INPUT_WIDTH, 0, 0] = 1
        batchPositionTensors[kth, 0:INPUT_WIDTH, INPUT_WIDTH - 1, 0] = 1
        # white occupied
        batchPositionTensors[kth, 0, 1:INPUT_WIDTH - 1, 1] = 1
        batchPositionTensors[kth, INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, 1] = 1

        board=np.ndarray(shape=(INPUT_WIDTH, INPUT_WIDTH), dtype=np.int32)
        RLTensorUtil.set_board(board, gamestate)

        turn = HexColor.BLACK
        for imove in gamestate:
            (x, y) = RLTensorUtil.intMoveToPair(imove)
            x, y = x + 1, y + 1
            ind = 0 if turn == HexColor.BLACK else 1
            batchPositionTensors[kth, x, y, ind] = 1
            batchPositionTensors[kth, x, y, INPUT_DEPTH - 1] = 0
            turn = HexColor.EMPTY - turn

        ind_bridge_black = 2
        ind_bridge_white = 3
        for i in xrange(INPUT_WIDTH - 1):
            for j in xrange(INPUT_WIDTH - 1):
                p1 = board[i, j], board[i + 1, j], board[i, j + 1], board[i + 1, j + 1]
                if p1[0] == HexColor.BLACK and p1[3] == HexColor.BLACK and p1[1] != HexColor.WHITE and p1[
                    2] != HexColor.WHITE:
                    batchPositionTensors[kth, i, j, ind_bridge_black] = 1
                    batchPositionTensors[kth, i + 1, j + 1, ind_bridge_black] = 1
                if p1[0] == HexColor.WHITE and p1[3] == HexColor.WHITE and p1[1] != HexColor.BLACK and p1[
                    2] != HexColor.BLACK:
                    batchPositionTensors[kth, i, j, ind_bridge_white] = 1
                    batchPositionTensors[kth, i + 1, j + 1, ind_bridge_white] = 1
                if j - 1 >= 0:
                    p2 = board[i, j], board[i + 1, j - 1], board[i + 1, j], board[i, j + 1]
                    if p2[1] == HexColor.BLACK and p2[3] == HexColor.BLACK and p2[0] != HexColor.WHITE and p2[
                        2] != HexColor.WHITE:
                        batchPositionTensors[kth, i + 1, j - 1, ind_bridge_black] = 1
                        batchPositionTensors[kth, i, j + 1, ind_bridge_black] = 1
                    if p2[1] == HexColor.WHITE and p2[3] == HexColor.WHITE and p2[0] != HexColor.BLACK and p2[
                        2] != HexColor.BLACK:
                        batchPositionTensors[kth, i + 1, j - 1, ind_bridge_white] = 1
                        batchPositionTensors[kth, i, j + 1, ind_bridge_white] = 1

    @staticmethod
    def set_board(board, gamestate):
        board.fill(0)
        board[0:INPUT_WIDTH, 0] = HexColor.BLACK
        board[0:INPUT_WIDTH, INPUT_WIDTH - 1] = HexColor.BLACK
        board[0, 1:INPUT_WIDTH - 1] = HexColor.WHITE
        board[INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1] = HexColor.WHITE
        turn = HexColor.BLACK
        for imove in gamestate:
            (x, y) = RLTensorUtil.intMoveToPair(imove)
            x, y = x + 1, y + 1
            board[x, y] = turn
            turn = HexColor.EMPTY - turn

    @staticmethod
    def intMoveToPair(imove):
        x=imove//BOARD_SIZE
        y=imove%BOARD_SIZE
        return (x,y)

