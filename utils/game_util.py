from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


from utils.commons import *
from zobrist.zobrist import HexColor

import numpy as np

NORTH_EDGE=-1
SOUTH_EDGE=-3

EAST_EDGE=-2
WEST_EDGE=-4

'''''
Rawmove in the format 'a10'
'''''
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

    @staticmethod
    def rotateMove180(intMove):
        assert(0<=intMove<BOARD_SIZE**2)
        return BOARD_SIZE**2 -1 - intMove

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

class RLTensorUtil13x13:
    def __init__(self):
        self._board = np.ndarray(dtype=np.int32, shape=(INPUT_WIDTH, INPUT_WIDTH))
        self.BSTONE_PLANE = 0
        self.WSTONE_PLANE = 1
        self.BBRIDGE_ENDPOINTS_PLANE = 2
        self.WBRIDGE_ENDPOINTS_PLANE = 3
        self.BTOPLAY_PLANE = 4
        self.WTOPLAY_PLANE = 5
        self.SAVE_BRIDGE_PLANE = 6
        self.FORM_BRIDGE_PLANE = 7
        self.EMPTY_POINTS_PLALNE = 8
        self.HISTORY_PLANE=9
        self.NUMPADDING = PADDINGS

    def set_position_label_in_batch(self, batchLabels, kth, nextMove):
        batchLabels[kth]=nextMove

    def set_position_tensors_in_batch(self, batchPositionTensors, kth, intState):
        # empty positions
        batchPositionTensors[kth, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING, self.EMPTY_POINTS_PLALNE] = 1

        batch_positions=batchPositionTensors
        # black occupied
        for i in range(self.NUMPADDING):
            batch_positions[kth, 0:INPUT_WIDTH, i, self.BSTONE_PLANE] = 1
            batch_positions[kth, 0:INPUT_WIDTH, INPUT_WIDTH - 1 - i, self.BSTONE_PLANE] = 1
        # white occupied
        for j in range(self.NUMPADDING):
            batch_positions[kth, j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING, self.WSTONE_PLANE] = 1
            batch_positions[kth, INPUT_WIDTH - 1 - j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING,
            self.WSTONE_PLANE] = 1

        self._set_board(intState)
        turn = HexColor.BLACK
        t=0.0
        # set black/white stone planes, and empty point plane
        for intMove in intState:
            (x, y) = MoveConvertUtil.intMoveToPair(intMove)
            x, y = x + self.NUMPADDING, y + self.NUMPADDING
            ind = self.BSTONE_PLANE if turn == HexColor.BLACK else self.WSTONE_PLANE
            batch_positions[kth, x, y, ind] = 1
            batch_positions[kth, x, y, self.EMPTY_POINTS_PLALNE] = 0

            #set history plane
            t +=1.0
            batch_positions[kth,x,y, self.HISTORY_PLANE]=np.exp(-1.0/t)
            turn = HexColor.EMPTY - turn

        # set toplay plane
        if turn == HexColor.BLACK:
            batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.BTOPLAY_PLANE] = 1
            batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.WTOPLAY_PLANE] = 0
        else:
            batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.BTOPLAY_PLANE] = 0
            batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.WTOPLAY_PLANE] = 1

        ind_bridge_black = self.BBRIDGE_ENDPOINTS_PLANE
        ind_bridge_white = self.WBRIDGE_ENDPOINTS_PLANE
        for i in xrange(self.NUMPADDING, INPUT_WIDTH - self.NUMPADDING):
            for j in xrange(self.NUMPADDING, INPUT_WIDTH - self.NUMPADDING):
                p1 = self._board[i, j], self._board[i + 1, j], self._board[i, j + 1], self._board[i + 1, j + 1]
                if p1[0] == HexColor.BLACK and p1[3] == HexColor.BLACK and p1[1] != HexColor.WHITE and p1[
                    2] != HexColor.WHITE:
                    batch_positions[kth, i, j, ind_bridge_black] = 1
                    batch_positions[kth, i + 1, j + 1, ind_bridge_black] = 1
                if p1[0] == HexColor.WHITE and p1[3] == HexColor.WHITE and p1[1] != HexColor.BLACK and p1[
                    2] != HexColor.BLACK:
                    batch_positions[kth, i, j, ind_bridge_white] = 1
                    batch_positions[kth, i + 1, j + 1, ind_bridge_white] = 1
                if j - 1 >= 0:
                    p2 = self._board[i, j], self._board[i + 1, j - 1], self._board[i + 1, j], self._board[i, j + 1]
                    if p2[1] == HexColor.BLACK and p2[3] == HexColor.BLACK and p2[0] != HexColor.WHITE and p2[
                        2] != HexColor.WHITE:
                        batch_positions[kth, i + 1, j - 1, ind_bridge_black] = 1
                        batch_positions[kth, i, j + 1, ind_bridge_black] = 1
                    if p2[1] == HexColor.WHITE and p2[3] == HexColor.WHITE and p2[0] != HexColor.BLACK and p2[
                        2] != HexColor.BLACK:
                        batch_positions[kth, i + 1, j - 1, ind_bridge_white] = 1
                        batch_positions[kth, i, j + 1, ind_bridge_white] = 1

                # for toplay save bridge
                if p1[0] == p1[3] and p1[0] == turn and p1[1] != turn and p1[2] != turn:
                    if p1[1] == HexColor.EMPTY and p1[2] == HexColor.EMPTY - turn:
                        batch_positions[kth, i + 1, j, self.SAVE_BRIDGE_PLANE] = 1
                    elif p1[1] == HexColor.EMPTY - turn and p1[2] == HexColor.EMPTY:
                        batch_positions[kth, i, j + 1, self.SAVE_BRIDGE_PLANE] = 1

                if j - 1 >= 0:
                    p2 = self._board[i, j], self._board[i + 1, j - 1], self._board[i + 1, j], self._board[i, j + 1]
                    if p2[1] == p2[3] and p2[1] == turn and p2[0] != turn and p2[2] != turn:
                        if p2[0] == HexColor.EMPTY and p2[2] == HexColor.EMPTY - turn:
                            batch_positions[kth, i, j, self.SAVE_BRIDGE_PLANE] = 1
                        elif p2[0] == HexColor.EMPTY - turn and p2[2] == HexColor.EMPTY:
                            batch_positions[kth, i + 1, j, self.SAVE_BRIDGE_PLANE] = 1

                # for toplay form bridge
                if p1[1] == HexColor.EMPTY and p1[2] == HexColor.EMPTY:
                    if p1[0] == HexColor.EMPTY and p1[3] == turn:
                        batch_positions[kth, i, j, self.FORM_BRIDGE_PLANE] = 1
                    elif p1[0] == turn and p1[3] == HexColor.EMPTY:
                        batch_positions[kth, i + 1, j + 1, self.FORM_BRIDGE_PLANE] = 1

                if j - 1 >= 0:
                    p2 = self._board[i, j], self._board[i + 1, j - 1], self._board[i + 1, j], self._board[i, j + 1]
                    if p2[0] == p2[2] == HexColor.EMPTY:
                        if p2[1] == HexColor.EMPTY and p2[3] == turn:
                            batch_positions[kth, i + 1, j - 1, self.FORM_BRIDGE_PLANE] = 1
                        elif p2[1] == turn and p2[3] == HexColor.EMPTY:
                            batch_positions[kth, i, j + 1, self.FORM_BRIDGE_PLANE] = 1

        #end
    '''A square board the same size as Tensor input, each point is either EMPTY, BLACK or WHITE
    used to check brige-related pattern,
    '''
    def _set_board(self, intState):
        self._board.fill(HexColor.EMPTY)
        # set black padding borders
        for i in range(self.NUMPADDING):
            self._board[0:INPUT_WIDTH, i] = HexColor.BLACK
            self._board[0:INPUT_WIDTH, INPUT_WIDTH - 1 - i] = HexColor.BLACK
        # set white padding borders
        for j in range(self.NUMPADDING):
            self._board[j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING] = HexColor.WHITE
            self._board[INPUT_WIDTH - 1 - j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING] = HexColor.WHITE
        turn = HexColor.BLACK
        for intMove in intState:
            (x, y) = MoveConvertUtil.intMoveToPair(intMove)
            x, y = x + self.NUMPADDING, y + self.NUMPADDING
            self._board[x, y] = turn
            turn = HexColor.EMPTY - turn
            # B[c3]=> c3 => ('c-'a')*boardsize+(3-1) , W[a11]=> a11

    def makeTensorInBatch(self, batchPositionTensors, batchLabels, kth, intState, intNextMove):
        self.set_position_label_in_batch(batchLabels, kth, intNextMove)
        self.set_position_tensors_in_batch(batchPositionTensors, kth, intState)
