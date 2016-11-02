
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from six.moves import xrange

import numpy as np

BOARD_SIZE = 8
BATCH_SIZE = 64
INPUT_WIDTH=BOARD_SIZE + 2
INPUT_DEPTH = 3

EVAL_BATCH_SIZE=5000

class data_util(object):

    def __init__(self, games=None, batch_size=None, batch_state=None, batch_label=None):
        self.games=games
        self.batchsize=batch_size
        self.batch_states=batch_state
        self.batch_labels=batch_label
        self.symmetry_checking=True

    def load_offline_data(self, dataset_location, train_data=True):
        self.dataset = dataset_location
        batchsize = BATCH_SIZE if train_data else EVAL_BATCH_SIZE
        self.batch_states = np.ndarray(shape=(batchsize, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), dtype=np.uint8)
        self.batch_labels = np.ndarray(shape=(batchsize,), dtype=np.uint8)
        self.batchsize = batchsize
        self.read_raw_data()

    def disable_symmetry_checking(self):
        self.symmetry_checking=False

    def convert(self, raw_game):
        single_game = [-1]  # initially, empty board, so set first move as -1
        moves = raw_game.split()
        for raw_move in moves:
            i = ord(raw_move[0].lower()) - ord('a')
            j = int("".join(raw_move[1:])) - 1
            single_game.append(i * BOARD_SIZE + j)
        return single_game

    def read_raw_data(self):
        self.games=[]
        with open(self.dataset, "r") as infile:
            for line in infile:
                self.games.append(self.convert(line))

    def check_symmetry(self, intgame):
        N  =BOARD_SIZE**2
        for i, intmove in enumerate(intgame):
            sym_move=N-1-intmove
            try:
                idx=intgame.index(sym_move)
            except ValueError:
                return False
            if idx % 2 != i % 2:
                return False
        return True

    def build_game_tobatch(self, kth, game_i, play_j):
        #empty positions
        self.batch_states[kth, 1:INPUT_WIDTH-1, 1:INPUT_WIDTH-1, INPUT_DEPTH - 1] = 1
        #black occupied
        self.batch_states[kth, 0:INPUT_WIDTH, 0, 0] = 1
        self.batch_states[kth, 0:INPUT_WIDTH, INPUT_WIDTH - 1, 0] = 1
        #white occupied
        self.batch_states[kth, 0, 1:INPUT_WIDTH - 1, 1] = 1
        self.batch_states[kth, INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, 1] = 1
        if(self.games[game_i][play_j] == -1):
            return
        # black plays first, the first channel for black
        turn = 0  # black is in 0-channel
        g= [self.games[game_i][j] for j in range(1,play_j+1,1)]

        is_symmetry = False if not self.symmetry_checking else self.check_symmetry(g)

        for move in g:
            x=move//BOARD_SIZE
            y=move%BOARD_SIZE
            self.batch_states[kth, x+1,y+1, turn]=1
            self.batch_states[kth, x+1,y+1, INPUT_DEPTH-1]=0
            turn = turn + 1
            turn = turn % (INPUT_DEPTH - 1)

        del g
        return is_symmetry

    def prepare_batch(self, offset1, offset2):
        k = 0
        new_offset1 = -1
        new_offset2 = -1
        self.batch_labels.fill(0)
        self.batch_states.fill(0)
        next_epoch=False
        while k < self.batchsize:
            for i in xrange(offset1, len(self.games)):
                assert(len(self.games[i]) > 1)
                for j in xrange(offset2, len(self.games[i]) - 1):
                    symmetry_board = self.build_game_tobatch(k, i, j)
                    self.batch_labels[k]=self.games[i][j + 1]
                    if(symmetry_board):
                        self.batch_labels[k]=min(self.batch_labels[k], BOARD_SIZE**2-1-self.batch_labels[k])
                    k = k + 1
                    if(k >= self.batchsize):
                        new_offset1 = i
                        new_offset2 = j + 1
                        break;
                offset2 = 0
                if(k >= self.batchsize):
                        break
            if(k < self.batchsize):
                next_epoch=True
                offset1 = 0
                offset2 = 0
        return (new_offset1, new_offset2, next_epoch)

if __name__ == "__main__":
    datatest=data_util()
    datatest.load_offline_data("data/7x7rawgames.dat", train_data=True)
    offset1 = 0
    offset2 = 0
    nepoch=0
    while(nepoch <= 1):
        o1, o2, next_epoch= datatest.prepare_batch(offset1, offset2)
        offset1 = o1
        offset2 = o2
        print("epoch", nepoch, "offset: ", o1, o2)
        if(next_epoch):
            nepoch += 1
