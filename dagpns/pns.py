from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from zobrist.zobrist import *
from utils.unionfind import *
from dagpns.node import Node
import copy

INF=200000000
BOARD_SIZE=3

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
            if (m2 in board and list(board).index(m2) % 2 == player-1):
                black_group.join(m2, intmove)
    else:

        if (x == 0):
            white_group.join(intmove, WEST_EDGE)
        if (x == BOARD_SIZE - 1):
            white_group.join(intmove, EAST_EDGE)

        for m in neighbors:
            im = m[0] * BOARD_SIZE + m[1]
            if (im in board and list(board).index(im) % 2 == player-1):
                white_group.join(im, intmove)
    # print(black_group.parent)
    return (black_group, white_group)

class PNS:
    def __init__(self):
        self.mWorkHash=None
        self.mTT=None
        self.mToplay=None
        self.workState=None
        self.zhash = ZobristHash(boardsize=BOARD_SIZE)
        self.node_cnt=None
        self.mid_calls=None

    def evaluate(self, moveseq):
        blackUF=unionfind()
        whiteUF=unionfind()
        toplay=HexColor.BLACK
        for m in moveseq:
            updateUF(moveseq, blackUF, whiteUF, m, toplay)
            toplay=HexColor.EMPTY - toplay
        outcome=winner(blackUF, whiteUF)
        if outcome!=HexColor.EMPTY:
            if(outcome!=self.mToplay):
                print("outcome ", outcome, "toplay: ", self.mToplay)
                print(moveseq)
            return outcome, True
        for m in range(BOARD_SIZE**2):
            if m not in moveseq:
                b,w=copy.deepcopy(blackUF), copy.deepcopy(whiteUF)
                moveseq.append(m)
                b,w=updateUF(moveseq, b,w,m,self.mToplay)
                res=winner(b,w)
                if res!=HexColor.EMPTY:
                    assert(res==self.mToplay)
                    return res, False
                moveseq.remove(m)
        return HexColor.EMPTY, False

    def dfpns(self,state, toplay):
        self.mToplay = toplay
        self.mWorkState=state
        self.rootToplay=toplay
        self.mTT={}
        self.node_cnt=self.mid_calls=0
        self.mWorkHash=self.zhash.get_hash(intstate=state)
        root = Node(phi=INF, delta=INF, code=self.mWorkHash)
        self.MID(root)
        if(root.delta>=INF):
            print(toplay, " Win")
        elif root.delta == 0:
            print(toplay, "Lose")
        else:
            print("Unknown, something wrong?")

        print("number nodes expanded: ", self.node_cnt)
        print("number of MID calls: ", self.mid_calls)

    def MID(self, n):
        print("MID call: ", self.mid_calls, "state: ", self.mWorkState, "toplay=", self.mToplay)
        assert(len(self.mWorkState)%2+1==self.mToplay)
        self.mid_calls +=1
        outcome, is_terminal=self.evaluate(self.mWorkState)
        if (outcome!=HexColor.EMPTY):
            if is_terminal == True:
                print("possible?")
                n.phi, n.delta = (0,INF) if outcome==self.mToplay else (INF, 0)
            if outcome==self.mToplay:
                (n.phi, n.delta)=(0, INF)
            else:
                (n.phi, n.delta)=(INF, 0)
            #print("Terminal state: ", self.mWorkState)
            #self.mToplay=HexColor.EMPTY - self.mToplay
            self.tt_write(n)
            return

        self.generate_moves()
        print("MID call2: ", self.mid_calls, "state: ", self.mWorkState, "toplay=", self.mToplay)

        phi_thre=self.deltaMin()
        delta_thre=self.phiSum()
        while n.phi > phi_thre and n.delta > delta_thre:
            c_best, delta2, best_move=self.selectChild()
            c_best.phi = n.delta - self.phiSum() + c_best.phi
            c_best.delta=min(n.phi, delta2+1)
            assert(best_move!=None)
            self.mWorkState.append(best_move)
            self.mWorkHash=self.zhash.update_hash(code=self.mWorkHash, intmove=best_move, intplayer=self.mToplay)
            self.mToplay= HexColor.EMPTY - self.mToplay
            print("workstate: ", self.mWorkState, "best_move", best_move, "toplay", self.mToplay)
            self.MID(c_best)
            print("before remove", best_move, self.mWorkState)
            self.mWorkState.remove(best_move)
            print("after remove", self.mWorkState)
            self.mToplay = HexColor.EMPTY - self.mToplay
            print("after work:", self.mWorkState, "toplay ", self.mToplay)
            self.mWorkHash = self.zhash.update_hash(code=self.mWorkHash, intmove=best_move, intplayer=self.mToplay)
            phi_thre=self.deltaMin()
            delta_thre=self.phiSum()
        n.phi=self.deltaMin()
        n.delta=self.phiSum()
        self.tt_write(n)

    #write new positions to TT
    def generate_moves(self):
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                child_code=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                pair=self.tt_lookup(child_code)
                if not pair:
                    print("save state: ", self.mWorkState, "move", i, "hash", child_code)
                    n=Node(code=child_code, phi=1, delta=1)
                    self.tt_write(n)
                    self.node_cnt += 1

    def tt_write(self, n):
        self.mTT[n.code]=(n.phi, n.delta)

    def tt_lookup(self, code):
        if code in self.mTT.keys():
            return self.mTT[code]
        else:
            return False

    def selectChild(self):
        delta_smallest=INF
        delta2=INF
        phi_best_child=INF
        best_child_code=None
        best_move=None
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                child_code=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                pair=self.tt_lookup(child_code)
                assert(pair)
                phi, delta=pair[0], pair[1]
                if delta < delta_smallest:
                    best_move=i
                    best_child_code=child_code
                    delta2=delta_smallest
                    delta_smallest=delta
                    phi_best_child=phi
                elif delta < delta2:
                    delta2 = delta
                if phi >= INF:
                    return Node(child_code, phi, delta), delta2, i

        return Node(best_child_code, phi_best_child, delta_smallest), delta2, best_move

    def phiSum(self):
        s=0
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                child_code=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                pair=self.tt_lookup(child_code)
                assert(pair)
                s+=pair[0]
        return s

    def deltaMin(self):
        min_delta=INF
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                child_code = self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                pair = self.tt_lookup(child_code)
                if not pair:
                    print(pair, child_code, self.mWorkState, "toplay ", self.mToplay, "move,", i)
                assert (pair)
                min_delta=min(min_delta, pair[1])
        return min_delta

if __name__ == "__main__":

    s=[2,0,1,3,4,7]
    b=unionfind()
    w=unionfind()
    p=HexColor.BLACK
    for i in s:
        updateUF(s,b,w,i,p)
        p=HexColor.EMPTY-p
    print("winner: ", winner(b,w))
    print(b.parent)
    print(b.rank)
    bcp=copy.deepcopy(b)
    bcp.parent[1]=-3
    print(b.parent)
    print(bcp.parent)
    pns=PNS()
    pns.mToplay=HexColor.BLACK
    print("lookead:", pns.evaluate(moveseq=s))
    state=[]
    pns.dfpns(state=state, toplay=HexColor.BLACK)
    pns2=PNS()
    for i in range(0):
        print("openning ", i)
        state=[i]
        pns2.dfpns(state=state, toplay=HexColor.WHITE)
