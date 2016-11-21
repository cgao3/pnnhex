from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from zobrist.zobrist import *
from utils.unionfind import *
from dagpns.commons import *
import time

import copy

INF=200000000

class DFPN:
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
                    moveseq.remove(m)
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
        print("1 for Black, 2 for White")
        if(root.delta>=INF):
            print(toplay, " Win")
        elif root.delta == 0:
            print(toplay, "Lose")
        else:
            print("Unknown, something wrong?")

        print("number nodes expanded: ", self.node_cnt, "MID calls:", self.mid_calls)

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
            self.tt_write(n)
            return

        self.generate_moves()

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
            self.MID(c_best)
            self.mWorkState.remove(best_move)
            self.mToplay = HexColor.EMPTY - self.mToplay
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

     pns2=DFPN()
     for i in range(BOARD_SIZE**2):
         start=time.time()
         print("\nopenning ", i)
         state=[i]
         toplay=HexColor.BLACK if len(state)%2==0 else HexColor.WHITE
         pns2.dfpns(state=state, toplay=toplay)
         end=time.time()
         print("time cost solving opening",i," :",(end-start))
