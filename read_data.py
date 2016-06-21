
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from six.moves import xrange

import numpy as np

games = []

BOARD_SIZE = 13
BATCH_SIZE = 64
INPUT_WIDTH=BOARD_SIZE + 2
INPUT_DEPTH = 3

batch_states = np.ndarray(shape=(BATCH_SIZE, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), dtype=np.uint8)
batch_labels = np.ndarray(shape=(BATCH_SIZE,), dtype=np.uint8)


def convert(raw_game):
    single_game = [-1]  # initially, empty board, so set first move as -1
    moves = raw_game.split()
    for raw_move in moves:
        i = ord(raw_move[0].lower()) - ord('a')
        j = int("".join(raw_move[1:])) - 1
        single_game.append(i * BOARD_SIZE + j)
    return single_game


def read_raw_data(dataset_name):
    with open(dataset_name, "r") as infile:
        for line in infile:
            games.append(convert(line))

def check_symmetry(intgame):
    N=BOARD_SIZE**2
    for i, intmove in enumerate(intgame):
        sym_move=N-1-intmove
        idx=-1
        try:
            idx=intgame.index(sym_move)
        except ValueError:
            return False
        if idx % 2 != i % 2:
            return False
    return True


def build_game_tobatch(kth, game_i, play_j):
    batch_states[kth, 1:INPUT_WIDTH-1, 1:INPUT_WIDTH-1, INPUT_DEPTH - 1] = 1
    if(games[game_i][play_j] == -1): 
        batch_states[kth,0:INPUT_WIDTH,0,0]=1
        batch_states[kth,0:INPUT_WIDTH,INPUT_WIDTH-1,0]=1
        batch_states[kth,0,1:INPUT_WIDTH-1,1]=1
        batch_states[kth,INPUT_WIDTH-1,1:INPUT_WIDTH-1,1]=1
        return
    # black plays first, the first channel for black
    turn = 0  # black is in 0-channel
    g= [games[game_i][j] for j in range(1,play_j+1,1)]

    is_symmetry = check_symmetry(g)

    for move in g:
        x=move//BOARD_SIZE
        y=move%BOARD_SIZE
        turn = turn % (INPUT_DEPTH - 1)
        batch_states[kth, x+1,y+1, turn]=1
        batch_states[kth, x+1,y+1, INPUT_DEPTH-1]=0
        if(turn==0):
            batch_states[kth,0:INPUT_WIDTH,0,0]=1
            batch_states[kth,0:INPUT_WIDTH,INPUT_WIDTH-1,0]=1
        else:
            batch_states[kth,0,1:INPUT_WIDTH-1,1]=1
            batch_states[kth,INPUT_WIDTH-1,1:INPUT_WIDTH-1,1]=1
        turn = turn + 1
    
    del g
    return is_symmetry


def prepare_batch(offset1, offset2):
    k = 0
    new_offset1 = -1
    new_offset2 = -1
    batch_labels.fill(0)
    batch_states.fill(0)
    next_epoch=False
    while k < BATCH_SIZE:
        for i in xrange(offset1, len(games)):
            assert(len(games[i]) > 1)
            for j in xrange(offset2, len(games[i]) - 1):
                symmetry_board = build_game_tobatch(k, i, j)
                batch_labels[k]=games[i][j + 1]
                if(symmetry_board):
                    batch_labels[k]=min(batch_labels[k], BOARD_SIZE*BOARD_SIZE-1-batch_labels[k])
                k = k + 1
                if(k >= BATCH_SIZE):
                    new_offset1 = i
                    new_offset2 = j + 1
                    break;
            offset2 = 0
            if(k >= BATCH_SIZE):
                    break
        if(k < BATCH_SIZE):
            next_epoch=True
            offset1 = 0
            offset2 = 0
    return (new_offset1, new_offset2, next_epoch)

if __name__ == "__main__":
    read_raw_data("data/train_games.dat")
    offset1 = 0
    offset2 = 0
    nepoch=0
    while(nepoch <= 1):
        o1, o2, next_epoch= prepare_batch(offset1, offset2)
        offset1 = o1
        offset2 = o2
        print("epoch", nepoch, "offset: ", o1, o2)
        if(next_epoch):
            nepoch += 1
        
# just a test
