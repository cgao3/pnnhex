
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from six.moves import xrange

import numpy as np

from zobrist.zobrist import *

BOARD_SIZE = 13
BATCH_SIZE = 64
PADDINGS=2
INPUT_WIDTH=BOARD_SIZE + 2*PADDINGS
INPUT_DEPTH = 9

EVAL_BATCH_SIZE=440

class ValueUtil(object):
    def __init__(self, srcStateValueFileName, batch_size):
        self.data_file_name=srcStateValueFileName
        self.batch_size=batch_size
        self.reader=open(self.data_file_name, "r")

        self.batch_positions=np.ndarray(shape=(batch_size, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), dtype=np.int32)
        self.batch_labels = np.ndarray(shape=(batch_size,), dtype=np.float32)
        self.currentLine = 0
        self._board = np.ndarray(dtype=np.int32, shape=(INPUT_WIDTH, INPUT_WIDTH))

    def close_file(self):
        self.reader.close()

    def prepare_batch(self):
        self.batch_positions.fill(0)
        self.batch_labels.fill(0)
        nextEpoch = False
        for i in xrange(self.batch_size):
            line = self.reader.readline()
            line = line.strip()
            if len(line) == 0:
                self.currentLine = 0
                self.reader.seek(0)
                line = self.reader.readline()
                nextEpoch = True
            self._build_batch_at(i, line)
            self.currentLine += 1
        return nextEpoch

    #only difference with position-action file is that the last number is +1/-1, while
    #position-action the last one is a move
    def _build_batch_at(self, kth, line):
        arr=line.strip().split()
        outcome=float(arr[-1]) # win or lose
        self.batch_labels[kth]=outcome
        assert(-0.001-1.0<outcome<1+0.0001)
        self.batch_positions[kth, 1:INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, INPUT_DEPTH - 1] = 1

        # black occupied
        self.batch_positions[kth, 0:INPUT_WIDTH, 0, 0] = 1
        self.batch_positions[kth, 0:INPUT_WIDTH, INPUT_WIDTH - 1, 0] = 1
        # white occupied
        self.batch_positions[kth, 0, 1:INPUT_WIDTH - 1, 1] = 1
        self.batch_positions[kth, INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, 1] = 1
        raws = arr[0:-1]
        self._set_board(raws)
        turn = HexColor.BLACK
        for raw in raws:
            (x, y) = self._toIntPair(raw)
            x, y = x + 1, y + 1
            ind = 0 if turn == HexColor.BLACK else 1
            self.batch_positions[kth, x, y, ind] = 1
            self.batch_positions[kth, x, y, INPUT_DEPTH - 1] = 0
            turn = HexColor.EMPTY - turn

        ind_bridge_black=2
        ind_bridge_white=3
        for i in xrange(INPUT_WIDTH-1):
            for j in xrange(INPUT_WIDTH-1):
                p1=self._board[i,j], self._board[i+1,j], self._board[i,j+1], self._board[i+1,j+1]
                if p1[0]==HexColor.BLACK and p1[3]==HexColor.BLACK and p1[1]!=HexColor.WHITE and p1[2]!=HexColor.WHITE:
                    self.batch_positions[kth,i,j,ind_bridge_black]=1
                    self.batch_positions[kth,i+1,j+1, ind_bridge_black]=1
                if p1[0]==HexColor.WHITE and p1[3]==HexColor.WHITE and p1[1]!=HexColor.BLACK and p1[2]!=HexColor.BLACK:
                    self.batch_positions[kth,i,j,ind_bridge_white]=1
                    self.batch_positions[kth,i+1,j+1,ind_bridge_white]=1
                if j-1>=0:
                    p2=self._board[i,j], self._board[i+1,j-1], self._board[i+1,j], self._board[i,j+1]
                    if p2[1] == HexColor.BLACK and p2[3] == HexColor.BLACK and p2[0] != HexColor.WHITE and p2[2] != HexColor.WHITE:
                        self.batch_positions[kth, i+1, j-1, ind_bridge_black] = 1
                        self.batch_positions[kth, i, j+1, ind_bridge_black] = 1
                    if p2[1] == HexColor.WHITE and p2[3] == HexColor.WHITE and p2[0] != HexColor.BLACK and p2[2] != HexColor.BLACK:
                        self.batch_positions[kth, i+1, j-1, ind_bridge_white] = 1
                        self.batch_positions[kth, i, j+1, ind_bridge_white] = 1


    def _set_board(self, raws):
        self._board.fill(0)
        self._board[0:INPUT_WIDTH,0]=HexColor.BLACK
        self._board[0:INPUT_WIDTH,INPUT_WIDTH-1]=HexColor.BLACK
        self._board[0,1:INPUT_WIDTH-1]=HexColor.WHITE
        self._board[INPUT_WIDTH-1, 1:INPUT_WIDTH-1]=HexColor.WHITE
        turn=HexColor.BLACK
        for raw in raws:
            (x,y)=self._toIntPair(raw)
            x,y=x+1,y+1
            self._board[x,y]=turn
            turn=HexColor.EMPTY-turn


    #B[c3]=> c3 => ('c-'a')*boardsize+(3-1) , W[a11]=> a11
    def _toIntPair(self, raw):
        x=ord(raw[2].lower())-ord('a')
        y=int(raw[3:-1])-1
        return (x,y)


#5 channels:
# black stone chanes, white stone channels, black bridge, white bridge, empty position channels
class PositionUtil3(object):
    def __init__(self, positiondata_filename, batch_size):
        self.data_file_name=positiondata_filename
        self.batch_size=batch_size
        self.reader=open(self.data_file_name, "r")
        self.batch_positions=np.ndarray(shape=(batch_size, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), dtype=np.uint32)
        self.batch_labels=np.ndarray(shape=(batch_size,), dtype=np.uint16)

        self.currentLine=0

        self._board=np.ndarray(dtype=np.int32, shape=(INPUT_WIDTH, INPUT_WIDTH))

    def close_file(self):
        self.reader.close()

    def prepare_batch(self):
        self.batch_positions.fill(0)
        self.batch_labels.fill(0)
        nextEpoch=False
        for i in xrange(self.batch_size):
            line=self.reader.readline()
            line=line.strip()
            if len(line)==0:
                self.currentLine=0
                self.reader.seek(0)
                line=self.reader.readline()
                nextEpoch=True
            self._build_batch_at(i, line)
            self.currentLine +=1
        return nextEpoch

    def _build_batch_at(self, kth, line):
        arr=line.strip().split()
        (x,y)=self._toIntPair(arr[-1])
        self.batch_labels[kth]=x*BOARD_SIZE+y

        self.batch_positions[kth, 1:INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, INPUT_DEPTH - 1] = 1
        # black occupied
        self.batch_positions[kth, 0:INPUT_WIDTH, 0, 0] = 1
        self.batch_positions[kth, 0:INPUT_WIDTH, INPUT_WIDTH - 1, 0] = 1
        # white occupied
        self.batch_positions[kth, 0, 1:INPUT_WIDTH - 1, 1] = 1
        self.batch_positions[kth, INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, 1] = 1
        raws = arr[0:-1]
        self._set_board(raws)
        turn = HexColor.BLACK
        for raw in raws:
            (x, y) = self._toIntPair(raw)
            x, y = x + 1, y + 1
            ind = 0 if turn == HexColor.BLACK else 1
            self.batch_positions[kth, x, y, ind] = 1
            self.batch_positions[kth, x, y, INPUT_DEPTH - 1] = 0
            turn = HexColor.EMPTY - turn

        ind_bridge_black=2
        ind_bridge_white=3
        for i in xrange(INPUT_WIDTH-1):
            for j in xrange(INPUT_WIDTH-1):
                p1=self._board[i,j], self._board[i+1,j], self._board[i,j+1], self._board[i+1,j+1]
                if p1[0]==HexColor.BLACK and p1[3]==HexColor.BLACK and p1[1]!=HexColor.WHITE and p1[2]!=HexColor.WHITE:
                    self.batch_positions[kth,i,j,ind_bridge_black]=1
                    self.batch_positions[kth,i+1,j+1, ind_bridge_black]=1
                if p1[0]==HexColor.WHITE and p1[3]==HexColor.WHITE and p1[1]!=HexColor.BLACK and p1[2]!=HexColor.BLACK:
                    self.batch_positions[kth,i,j,ind_bridge_white]=1
                    self.batch_positions[kth,i+1,j+1,ind_bridge_white]=1
                if j-1>=0:
                    p2=self._board[i,j], self._board[i+1,j-1], self._board[i+1,j], self._board[i,j+1]
                    if p2[1] == HexColor.BLACK and p2[3] == HexColor.BLACK and p2[0] != HexColor.WHITE and p2[2] != HexColor.WHITE:
                        self.batch_positions[kth, i+1, j-1, ind_bridge_black] = 1
                        self.batch_positions[kth, i, j+1, ind_bridge_black] = 1
                    if p2[1] == HexColor.WHITE and p2[3] == HexColor.WHITE and p2[0] != HexColor.BLACK and p2[2] != HexColor.BLACK:
                        self.batch_positions[kth, i+1, j-1, ind_bridge_white] = 1
                        self.batch_positions[kth, i, j+1, ind_bridge_white] = 1

    def _set_board(self, raws):
        self._board.fill(0)
        self._board[0:INPUT_WIDTH,0]=HexColor.BLACK
        self._board[0:INPUT_WIDTH,INPUT_WIDTH-1]=HexColor.BLACK
        self._board[0,1:INPUT_WIDTH-1]=HexColor.WHITE
        self._board[INPUT_WIDTH-1, 1:INPUT_WIDTH-1]=HexColor.WHITE
        turn=HexColor.BLACK
        for raw in raws:
            (x,y)=self._toIntPair(raw)
            x,y=x+1,y+1
            self._board[x,y]=turn
            turn=HexColor.EMPTY-turn


    #B[c3]=> c3 => ('c-'a')*boardsize+(3-1) , W[a11]=> a11
    def _toIntPair(self, raw):
        x=ord(raw[2].lower())-ord('a')
        y=int(raw[3:-1])-1
        return (x,y)

class PositionUtil9(object):
    '''
    input depth =9,
    black stones, white stones, black bridge endpoints, white bridge endpoints,
    black toplay planes, white toplay planes // of those two planes, only one is filled with 1.
    toplay(black or white) savebridge points, toplay form bridges, toplay empty points
    '''
    def __init__(self, positiondata_filename, batch_size):
        assert(INPUT_DEPTH==9)
        self.data_file_name = positiondata_filename
        self.batch_size = batch_size
        self.reader = open(self.data_file_name, "r")
        self.batch_positions = np.ndarray(shape=(batch_size, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), dtype=np.uint32)
        self.batch_labels = np.ndarray(shape=(batch_size,), dtype=np.uint16)
        self.currentLine = 0
        self._board = np.ndarray(dtype=np.int32, shape=(INPUT_WIDTH, INPUT_WIDTH))

        self.BSTONE_PLANE=0
        self.WSTONE_PLANE=1
        self.BBRIDGE_ENDPOINTS_PLANE=2
        self.WBRIDGE_ENDPOINTS_PLANE=3
        self.BTOPLAY_PLANE=4
        self.WTOPLAY_PLANE=5
        self.SAVE_BRIDGE_PLANE=6
        self.FORM_BRIDGE_PLANE=7
        self.EMPTY_POINTS_PLALNE=8

        self.NUMPADDING=PADDINGS

        assert(INPUT_WIDTH == BOARD_SIZE + 2 * self.NUMPADDING)

    def close_file(self):
        self.reader.close()

    def prepare_batch(self):
        self.batch_positions.fill(0)
        self.batch_labels.fill(0)
        nextEpoch = False
        for i in xrange(self.batch_size):
            line = self.reader.readline()
            line = line.strip()
            if len(line) == 0:
                self.currentLine = 0
                self.reader.seek(0)
                line = self.reader.readline()
                nextEpoch = True
            self._build_batch_at(i, line)
            self.currentLine += 1
        return nextEpoch

    def _build_batch_at(self, kth, line):
        self.flagFlip = False
        arr = line.strip().split()
        (x, y) = self._toIntPair(arr[-1])

        self.batch_labels[kth] = x * BOARD_SIZE + y

        #empty positions
        self.batch_positions[kth, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING, self.EMPTY_POINTS_PLALNE] = 1
        # black occupied
        for i in range(self.NUMPADDING):
            self.batch_positions[kth, 0:INPUT_WIDTH, i, self.BSTONE_PLANE] = 1
            self.batch_positions[kth, 0:INPUT_WIDTH, INPUT_WIDTH - 1 - i, self.BSTONE_PLANE] = 1
        # white occupied
        for j in range(self.NUMPADDING):
            self.batch_positions[kth, j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING, self.WSTONE_PLANE] = 1
            self.batch_positions[kth, INPUT_WIDTH - 1-j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING, self.WSTONE_PLANE] = 1
        raws = arr[0:-1]
        self._set_board(raws)
        turn = HexColor.BLACK
        #set black/white stone planes, and empty point plane
        for raw in raws:
            (x, y) = self._toIntPair(raw)
            x, y = x + self.NUMPADDING, y + self.NUMPADDING
            ind = self.BSTONE_PLANE if turn == HexColor.BLACK else self.WSTONE_PLANE
            self.batch_positions[kth, x, y, ind] = 1
            self.batch_positions[kth, x, y, self.EMPTY_POINTS_PLALNE] = 0
            turn = HexColor.EMPTY - turn

        #set toplay plane
        if turn==HexColor.BLACK:
            self.batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.BTOPLAY_PLANE]=1
            self.batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.WTOPLAY_PLANE]=0
        else:
            self.batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.BTOPLAY_PLANE] = 0
            self.batch_positions[kth, 0:INPUT_WIDTH, 0:INPUT_WIDTH, self.WTOPLAY_PLANE] = 1

        ind_bridge_black = self.BBRIDGE_ENDPOINTS_PLANE
        ind_bridge_white = self.WBRIDGE_ENDPOINTS_PLANE
        for i in xrange(self.NUMPADDING, INPUT_WIDTH - self.NUMPADDING):
            for j in xrange(self.NUMPADDING, INPUT_WIDTH - self.NUMPADDING):
                p1 = self._board[i, j], self._board[i + 1, j], self._board[i, j + 1], self._board[i + 1, j + 1]
                if p1[0] == HexColor.BLACK and p1[3] == HexColor.BLACK and p1[1] != HexColor.WHITE and p1[
                    2] != HexColor.WHITE:
                    self.batch_positions[kth, i, j, ind_bridge_black] = 1
                    self.batch_positions[kth, i + 1, j + 1, ind_bridge_black] = 1
                if p1[0] == HexColor.WHITE and p1[3] == HexColor.WHITE and p1[1] != HexColor.BLACK and p1[
                    2] != HexColor.BLACK:
                    self.batch_positions[kth, i, j, ind_bridge_white] = 1
                    self.batch_positions[kth, i + 1, j + 1, ind_bridge_white] = 1
                if j - 1 >= 0:
                    p2 = self._board[i, j], self._board[i + 1, j - 1], self._board[i + 1, j], self._board[i, j + 1]
                    if p2[1] == HexColor.BLACK and p2[3] == HexColor.BLACK and p2[0] != HexColor.WHITE and p2[
                        2] != HexColor.WHITE:
                        self.batch_positions[kth, i + 1, j - 1, ind_bridge_black] = 1
                        self.batch_positions[kth, i, j + 1, ind_bridge_black] = 1
                    if p2[1] == HexColor.WHITE and p2[3] == HexColor.WHITE and p2[0] != HexColor.BLACK and p2[
                        2] != HexColor.BLACK:
                        self.batch_positions[kth, i + 1, j - 1, ind_bridge_white] = 1
                        self.batch_positions[kth, i, j + 1, ind_bridge_white] = 1

                #for toplay save bridge
                if p1[0] == p1[3] and p1[0]==turn and p1[1]!=turn and p1[2]!=turn:
                    if p1[1]==HexColor.EMPTY and p1[2]==HexColor.EMPTY-turn:
                        self.batch_positions[kth, i+1, j, self.SAVE_BRIDGE_PLANE]=1
                    elif p1[1]==HexColor.EMPTY-turn and p1[2]==HexColor.EMPTY:
                        self.batch_positions[kth, i, j+1, self.SAVE_BRIDGE_PLANE]=1

                if j-1 >=0:
                    p2 = self._board[i, j], self._board[i + 1, j - 1], self._board[i + 1, j], self._board[i, j + 1]
                    if p2[1] == p2[3] and p2[1] == turn and p2[0] != turn and p2[2] != turn:
                        if p2[0] == HexColor.EMPTY and p2[2] == HexColor.EMPTY - turn:
                            self.batch_positions[kth, i , j, self.SAVE_BRIDGE_PLANE] = 1
                        elif p2[0] == HexColor.EMPTY - turn and p2[2] == HexColor.EMPTY:
                            self.batch_positions[kth, i+1, j, self.SAVE_BRIDGE_PLANE] = 1

                #for toplay form bridge
                if  p1[1]==HexColor.EMPTY and p1[2]==HexColor.EMPTY:
                    if p1[0] == HexColor.EMPTY and p1[3] == turn:
                        self.batch_positions[kth, i, j, self.FORM_BRIDGE_PLANE] = 1
                    elif p1[0] == turn and p1[3] == HexColor.EMPTY:
                        self.batch_positions[kth, i+1, j + 1, self.FORM_BRIDGE_PLANE] = 1

                if j - 1 >= 0:
                    p2 = self._board[i, j], self._board[i + 1, j - 1], self._board[i + 1, j], self._board[i, j + 1]
                    if p2[0] == p2[2] == HexColor.EMPTY:
                        if p2[1] == HexColor.EMPTY and p2[3] ==  turn:
                            self.batch_positions[kth, i+1, j-1, self.FORM_BRIDGE_PLANE] = 1
                        elif p2[1] == turn and p2[3] == HexColor.EMPTY:
                            self.batch_positions[kth, i, j+1, self.FORM_BRIDGE_PLANE] = 1

    '''A square board the same size as Tensor input, each point is either EMPTY, BLACK or WHITE
    used to check brige-related pattern,
    '''
    def _set_board(self, raws):
        self._board.fill(HexColor.EMPTY)
        #set black padding borders
        for i in range(self.NUMPADDING):
            self._board[0:INPUT_WIDTH, i] = HexColor.BLACK
            self._board[0:INPUT_WIDTH, INPUT_WIDTH - 1-i] = HexColor.BLACK
        #set white padding borders
        for j in range(self.NUMPADDING):
            self._board[j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING] = HexColor.WHITE
            self._board[INPUT_WIDTH - 1 -j, self.NUMPADDING:INPUT_WIDTH - self.NUMPADDING] = HexColor.WHITE
        turn = HexColor.BLACK
        for raw in raws:
            (x, y) = self._toIntPair(raw)
            x, y = x + self.NUMPADDING, y + self.NUMPADDING
            self._board[x, y] = turn
            turn = HexColor.EMPTY - turn
            # B[c3]=> c3 => ('c-'a')*boardsize+(3-1) , W[a11]=> a11

    def _toIntPair(self, raw):
        x = ord(raw[2].lower()) - ord('a')
        y = int(raw[3:-1]) - 1
        assert(0<=x<BOARD_SIZE and 0<=y<BOARD_SIZE)
        imove=x*BOARD_SIZE+y
        if self.flagFlip:
            imove=BOARD_SIZE**2 - imove
        x=imove//BOARD_SIZE
        y=imove%BOARD_SIZE
        return (x, y)

#81 channels.
class PositionUtil81(object):
    def __init__(self, positiondata_filename, batch_size):
        self.data_file_name=positiondata_filename
        self.batch_size=batch_size
        self.reader=open(self.data_file_name, "r")
        self.batch_positions=np.ndarray(shape=(batch_size, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), dtype=np.uint32)
        self.batch_labels=np.ndarray(shape=(batch_size,), dtype=np.uint16)

        self.currentLine=0

        self._board=np.ndarray(dtype=np.int32, shape=(INPUT_WIDTH, INPUT_WIDTH))


    def close_file(self):
        self.reader.close()

    def prepare_batch(self):
        self.batch_positions.fill(0)
        self.batch_labels.fill(0)
        nextEpoch=False
        for i in xrange(self.batch_size):
            line=self.reader.readline()
            line=line.strip()
            if len(line)==0:
                self.currentLine=0
                self.reader.seek(0)
                line=self.reader.readline()
                nextEpoch=True
            self._build_batch_at(i, line)
            self.currentLine +=1
        return nextEpoch

    def _build_batch_at(self, kth, line):
        arr=line.strip().split()
        (x,y)=self._toIntPair(arr[-1])
        self.batch_labels[kth]=x*BOARD_SIZE+y

        raws=arr[0:-1]
        self._set_board(raws)
        for i in xrange(INPUT_WIDTH-1):
            for j in xrange(INPUT_WIDTH-1):
                p1=self._board[i,j], self._board[i+1,j], self._board[i,j+1], self._board[i+1,j+1]
                pos=p1[0]*27+p1[1]*9+p1[2]*3+p1[3]
                self.batch_positions[kth,i,j,pos]=1
                self.batch_positions[kth, i+1, j, pos] = 1
                self.batch_positions[kth, i, j+1, pos] = 1
                self.batch_positions[kth, i+1, j+1, pos] = 1

    def _set_board(self, raws):
        self._board.fill(0)
        self._board[0:INPUT_WIDTH,0]=HexColor.BLACK
        self._board[0:INPUT_WIDTH,INPUT_WIDTH-1]=HexColor.BLACK
        self._board[0,1:INPUT_WIDTH-1]=HexColor.WHITE
        self._board[INPUT_WIDTH-1, 1:INPUT_WIDTH-1]=HexColor.WHITE
        turn=HexColor.BLACK
        for raw in raws:
            (x,y)=self._toIntPair(raw)
            x,y=x+1,y+1
            self._board[x,y]=turn
            turn=HexColor.EMPTY-turn


    #B[c3]=> c3 => ('c-'a')*boardsize+(3-1) , W[a11]=> a11
    def _toIntPair(self, raw):
        x=ord(raw[2].lower())-ord('a')
        y=int(raw[3:-1])-1
        return (x,y)


#this class is for Supervised Learning
#positions in the format B[a1] W[a2]
class PositionUtil(object):
    def __init__(self, positiondata_filename, batch_size):
        self.data_file_name=positiondata_filename
        self.batch_size=batch_size
        self.reader=open(self.data_file_name, "r")
        self.batch_positions=np.ndarray(shape=(batch_size, INPUT_WIDTH, INPUT_WIDTH, INPUT_DEPTH), dtype=np.uint32)
        self.batch_labels=np.ndarray(shape=(batch_size,), dtype=np.uint16)

        self.currentLine=0

    def close_file(self):
        self.reader.close()

    def prepare_batch(self):
        self.batch_positions.fill(0)
        self.batch_labels.fill(0)
        nextEpoch=False
        for i in xrange(self.batch_size):
            line=self.reader.readline()
            line=line.strip()
            if len(line)==0:
                self.currentLine=0
                self.reader.seek(0)
                line=self.reader.readline()
                nextEpoch=True
            self._build_batch_at(i, line)
            self.currentLine +=1
        return nextEpoch

    def _build_batch_at(self, kth, line):
        arr=line.strip().split()
        (x,y)=self._toIntPair(arr[-1])
        self.batch_labels[kth]=x*BOARD_SIZE+y
        self.batch_positions[kth, 1:INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, INPUT_DEPTH - 1] = 1
        # black occupied
        self.batch_positions[kth, 0:INPUT_WIDTH, 0, 0] = 1
        self.batch_positions[kth, 0:INPUT_WIDTH, INPUT_WIDTH - 1, 0] = 1
        # white occupied
        self.batch_positions[kth, 0, 1:INPUT_WIDTH - 1, 1] = 1
        self.batch_positions[kth, INPUT_WIDTH - 1, 1:INPUT_WIDTH - 1, 1] = 1
        raws=arr[0:-1]
        turn=HexColor.BLACK
        for raw in raws:
            (x,y)=self._toIntPair(raw)
            x,y=x+1,y+1
            ind=0 if turn==HexColor.BLACK else 1
            self.batch_positions[kth,x,y, ind]=1
            self.batch_positions[kth,x,y, INPUT_DEPTH-1]=0
            turn=HexColor.EMPTY-turn

    #B[c3]=> c3 => ('c-'a')*boardsize+(3-1) , W[a11]=> a11
    def _toIntPair(self, raw):
        x=ord(raw[2].lower())-ord('a')
        y=int(raw[3:-1])-1
        return (x,y)

#this class is for reinforcement learning
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
                        break
                offset2 = 0
                if(k >= self.batchsize):
                        break
            if(k < self.batchsize):
                next_epoch=True
                offset1 = 0
                offset2 = 0
        return (new_offset1, new_offset2, next_epoch)

if __name__ == "__main__":
    datatest=PositionUtil(positiondata_filename="data/8x8/positions1.txt", batch_size=BATCH_SIZE)
    nextEpoch=False
    while nextEpoch==False:
        nextEpoch=datatest.prepare_batch()
        print("offset ", datatest.reader.tell(), "line:", datatest.currentLine)
