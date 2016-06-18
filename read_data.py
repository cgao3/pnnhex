
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

games = []

BOARD_SIZE = 13
BATCH_SIZE = 64
INPUT_WIDTH=BOARD_SIZE + 2
INPUT_DEPTH = 3
nEpoch = 0

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

def symmetry_move(int_move):
    N = BOARD_SIZE * BOARD_SIZE
    return N - 1 - int_move

def rotate180(int_game):
    g2 = []
    for i in int_game:
        if(i == -1):
            g2.append(i)
        else:
            g2.append(symmetry_move(i))
    return g2
    
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
    g=[]
    for j in xrange(1,play_j+1):
        g.append(games[game_i][j])
    g2=rotate180(g)
    g=min(g,g2)
    for move in g:
        x=move//BOARD_SIZE
        y=move%BOARD_SIZE
        turn = turn % (INPUT_DEPTH - 1)
        batch_states[kth, x+1,y+1, turn]=1
        batch_states[kth, x+1,y+1, INPUT_DEPTH-1]=0
        #batch_states[kth, ind // BOARD_SIZE, ind % BOARD_SIZE, turn] = 1
        #batch_states[kth, ind // BOARD_SIZE, ind % BOARD_SIZE, INPUT_DEPTH - 1] = 0  # position occupied
        if(turn==0):
            batch_states[kth,0:INPUT_WIDTH,0,0]=1
            batch_states[kth,0:INPUT_WIDTH,INPUT_WIDTH-1,0]=1
        else:
            batch_states[kth,0,1:INPUT_WIDTH-1,1]=1
            batch_states[kth,INPUT_WIDTH-1,1:INPUT_WIDTH-1,1]=1
        turn = turn + 1
    
    del g,g2    
            
def prepare_batch(offset1, offset2):
    k = 0
    global nEpoch
    new_offset1 = -1
    new_offset2 = -1
    while k < BATCH_SIZE:
        for i in xrange(offset1, len(games)):
            assert(len(games[i]) > 1)
            for j in xrange(offset2, len(games[i]) - 1):
                build_game_tobatch(k, i, j)
                batch_labels[k]=games[i][j + 1] #] = 1
                k = k + 1
                if(k >= BATCH_SIZE):
                    new_offset1 = i
                    new_offset2 = j + 1
                    break;
            offset2 = 0
            if(k >= BATCH_SIZE):
                    break
        if(k < BATCH_SIZE):
            nEpoch += 1
            offset1 = 0
            offset2 = 0
    return (new_offset1, new_offset2)

if __name__ == "__main__":
    read_raw_data("data/train_games.dat")
    offset1 = 0
    offset2 = 0
    while(nEpoch <= 1):    
        o1, o2 = prepare_batch(offset1, offset2)
        offset1 = o1
        offset2 = o2
        print(o1, o2)
        
# just a test
