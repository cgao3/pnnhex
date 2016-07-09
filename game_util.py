from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from read_data import BOARD_SIZE
from read_data import INPUT_DEPTH, INPUT_WIDTH

import numpy as np

NORTH_EDGE=-1
SOUTH_EDGE=-3

EAST_EDGE=-2
WEST_EDGE=-4

def int_to_pair(intmove):
    x=intmove//BOARD_SIZE
    y=intmove%BOARD_SIZE
    return (x,y)

def pair_to_int(move):
    x,y=move[0],move[1]
    return x*BOARD_SIZE+y

def raw_move_to_int(raw_move):
    x=ord(raw_move[0].lower())-ord('a')
    y=int(raw_move[1:])-1
    return x*BOARD_SIZE+y

def raw_move_to_pair(raw_move):
    x=ord(raw_move[0].lower())-ord('a')
    y=int(raw_move[1:])-1
    return (x,y)

def intmove_to_raw(intmove):
    x,y=int_to_pair(intmove)
    y +=1
    return chr(x+ord('a'))+repr(y)

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

def update_tensor(tensor, player, intmove):
    x, y = intmove // BOARD_SIZE + 1, intmove % BOARD_SIZE + 1
    tensor[0, x, y, player] = 1
    tensor[0, x, y, 2] = 0
    return tensor

def make_empty_board_tensor(tensor):
    tensor.fill(0)
    # black occupied
    tensor[0, 0:INPUT_WIDTH, 0, 0] = 1
    tensor[0, 0:INPUT_WIDTH, INPUT_WIDTH - 1, 0] = 1
    # white occupied
    tensor[0, 0, 1:INPUT_WIDTH - 1, 1] = 1
    tensor[0, INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, 1] = 1
    # empty positions
    tensor[0, 1:INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, INPUT_DEPTH - 1] = 1


#0 for black win, 1-white win, -1 unsettled.
def winner(black_group, white_group):
    if (black_group.connected(NORTH_EDGE, SOUTH_EDGE)):
        return 0
    elif (white_group.connected(WEST_EDGE, EAST_EDGE)):
        return 1
    else:
        return -1

#player either 0 or 1, 0-black, 1-white
def update_unionfind(intmove, player, board, black_group, white_group):
    x, y = intmove // BOARD_SIZE, intmove % BOARD_SIZE
    neighbors = []
    pattern = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, 1), (1, -1)]
    for p in pattern:
        x1, y1 = p[0] + x, p[1] + y
        if 0 <= x1 < BOARD_SIZE and 0 <= y1 < BOARD_SIZE:
            neighbors.append((x1, y1))
    if (player == 0):

        if (y == 0):
            black_group.join(intmove, NORTH_EDGE)
        if (y == BOARD_SIZE - 1):
            black_group.join(intmove, SOUTH_EDGE)

        for m in neighbors:
            m2 = m[0] * BOARD_SIZE + m[1]
            if (m2 in board and list(board).index(m2) % 2 == player):
                black_group.join(m2, intmove)
    else:

        if (x == 0):
            white_group.join(intmove, WEST_EDGE)
        if (x == BOARD_SIZE - 1):
            white_group.join(intmove, EAST_EDGE)

        for m in neighbors:
            im = m[0] * BOARD_SIZE + m[1]
            if (im in board and list(board).index(im) % 2 == player):
                white_group.join(im, intmove)
    # print(black_group.parent)
    return (black_group, white_group)


# input is raw score such as [-20,30,10]
def softmax_selection(logits, currentstate):
    logits = np.squeeze(logits)
    empty_positions = [i for i in range(BOARD_SIZE ** 2) if i not in currentstate]
    # print("empty positions:", empty_positions)
    # print(logits)
    effective_logits = [logits[i] for i in empty_positions]
    max_value = np.max(effective_logits)
    effective_logits = effective_logits - max_value
    effective_logits = np.exp(effective_logits)
    sum_value = np.sum(effective_logits)
    prob = effective_logits / sum_value
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