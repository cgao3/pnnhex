from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from zobrist.zobrist import *
from utils.unionfind import *
from dagpns.node import Node
import copy

INF=200000000
BOARD_SIZE=4

NORTH_EDGE=-1
SOUTH_EDGE=-2
WEST_EDGE=-3
EAST_EDGE=-4

EPSILON=1e-5

#0 for black win, 1-white win, -1 unsettled.
def winner(black_group, white_group):
    if (black_group.connected(NORTH_EDGE, SOUTH_EDGE)):
        return HexColor.BLACK
    elif (white_group.connected(WEST_EDGE, EAST_EDGE)):
        return HexColor.WHITE
    else:
        return HexColor.EMPTY

def updateUF(board, black_group, white_group, intmove, player):
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

class PNS:
    def __init__(self):
        self.mWorkHash=0
        #self.mTT=np.ndarray(shape=(2**21), dtype='2int32')
        self.mTT={}
        #self.mTT.fill(-1.0)
        self.mToplay=None
        self.workState=None
        self.zhash = ZobristHash(boardsize=BOARD_SIZE)
        self.node_cnt=0
        self.mid_calls=0

    def evaluate(self, moveseq):
        blackUF=unionfind()
        whiteUF=unionfind()
        toplay=HexColor.BLACK
        for m in moveseq:
            updateUF(moveseq, blackUF, whiteUF, m, toplay)
            toplay=HexColor.EMPTY - toplay
        outcome=winner(blackUF, whiteUF)
        if outcome!=HexColor.EMPTY:
            return outcome
        for m in range(BOARD_SIZE**2):
            if m not in moveseq:
                b,w=copy.deepcopy(blackUF), copy.deepcopy(whiteUF)
                moveseq.append(m)
                b,w=updateUF(moveseq, b,w,m,self.mToplay)
                res=winner(b,w)
                if res!=HexColor.EMPTY:
                    return res
                moveseq.remove(m)
        return winner(blackUF, whiteUF)

    def dfpns(self,state, toplay):
        self.mToplay = toplay
        self.mWorkState=state
        self.rootToplay=toplay
        self.mTT={}
        self.mWorkHash=self.zhash.get_hash(intstate=state)
        root = Node(phi=INF, delta=INF, code=self.zhash.m_hash)
        self.MID(root)
        if(root.phi==0):
            print(toplay, " Win")
        elif root.delta==0:
            print(toplay, "Lose")
        else:
            print("Unknown, something wrong?")

        print("number nodes expanded: ", self.node_cnt)
        print("number of MID calls: ", self.mid_calls)

    def MID(self, n):
        self.mid_calls +=1
        outcome=self.evaluate(self.mWorkState)
        if (outcome!=HexColor.EMPTY):
            if outcome==self.mToplay:
                (n.phi, n.delta)=(INF, 0)
            else:
                (n.phi, n.delta)=(0, INF)
            print("Terminal state: ", self.mWorkState)
            self.tt_write(n)
            return

        self.generate_moves()
        while n.phi > self.deltaMin() and n.delta > self.phiSum():
            c_best, delta2, best_move=self.selectChild()
            c_best.phi = n.delta - self.phiSum() + c_best.phi
            c_best.delta=min(n.phi, delta2+1)
            self.mWorkState.append(best_move)
            self.mWorkHash=self.zhash.update_hash(code=self.mWorkHash, intmove=best_move, intplayer=self.mToplay)
            self.mToplay= HexColor.EMPTY - self.mToplay
            self.MID(c_best)
            self.mWorkState.remove(best_move)
            self.mToplay = HexColor.EMPTY - self.mToplay
            self.mWorkHash = self.zhash.update_hash(code=self.mWorkHash, intmove=best_move, intplayer=self.mToplay)
        n.phi=self.deltaMin()
        n.delta=self.phiSum()
        self.tt_write(n)

    #write new positions to TT
    def generate_moves(self):
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                tcode=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                leaf=self.tt_lookup(tcode)
                if leaf==False:
                    n=Node(code=tcode, phi=1, delta=1)
                    self.tt_write(n)
                    self.node_cnt += 1


    def tt_write(self, n):
        self.mTT[n.code]=(n.phi, n.delta)

    def tt_lookup(self, code):
        if code in self.mTT.keys():
            return self.mTT[code]
        else:
            #no this node
            return False

    def selectChild(self):
        delta1=INF
        delta2=INF
        phi1=INF
        best_child=None
        best_move=None
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                tcode=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                pair=self.tt_lookup(tcode)
                if pair ==False: pair=(1,1)
                phi, delta=pair[0], pair[1]
                if delta < delta1:
                    best_move=i
                    best_child=tcode
                    delta2=delta1
                    delta1=delta
                    phi1=phi
                elif delta < delta2:
                    delta2 = delta
                if phi >= INF:
                    return Node(tcode, phi, delta), delta2

        return Node(best_child, phi1, delta1), delta2, best_move

    def phiSum(self):
        s=0
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                tcode=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                pair=self.tt_lookup(tcode)
                if pair == False:
                    pair=(1,1)
                s+=pair[0]
                assert (pair)
        return s

    def deltaMin(self):
        min_delta=INF
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                tcode = self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                pair = self.tt_lookup(tcode)
                if pair ==False:
                    pair=(1,1)
                assert (pair)
                min_delta=min(min_delta, pair[1])
        return min_delta

if __name__ == "__main__":
    pns=PNS()
    state=[]
    print("hello")
    pns.dfpns(state=state, toplay=HexColor.BLACK)
