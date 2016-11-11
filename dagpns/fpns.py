from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from zobrist.zobrist import *
from utils.unionfind import *
from dagpns.node import Node
import copy
from dagpns.pns import winner, updateUF
import Queue

INF=200000000.0
BOARD_SIZE=4

NORTH_EDGE=-1
SOUTH_EDGE=-2
WEST_EDGE=-3
EAST_EDGE=-4

EPSILON=1e-5

class FPNS:
    def __init__(self):
        self.mWorkHash=0
        self.mTT={}
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

    def fdfpns(self,state, toplay):
        self.mToplay = toplay
        self.mWorkState=state
        self.rootToplay=toplay
        self.mTT={}
        self.mWorkHash=self.zhash.get_hash(intstate=state)
        root = Node(phi=INF, delta=INF, code=self.zhash.m_hash, parents=[])
        self.MID(root)
        if(abs(root.phi) < EPSILON):
            print(toplay, " Win")
        elif abs(root.delta) <EPSILON:
            print(toplay, "Lose")
        else:
            print("Unknown, something wrong?")

        print("number nodes expanded: ", self.node_cnt)
        print("number of MID calls, ", self.mid_calls)

    def MID(self, n):
        self.mid_calls +=1
        print("MID calls, ", self.mid_calls)
        outcome=self.evaluate(self.mWorkState)
        if (outcome!=HexColor.EMPTY):
            if outcome==self.mToplay:
                (n.phi, n.delta)=(INF, 0.)
            else:
                (n.phi, n.delta)=(0., INF)
            print("Terminal state: ", self.mWorkState)

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
                if leaf == False:
                    n=Node(code=tcode, phi=1., delta=1., parents=[self.mWorkHash])
                    self.tt_write(n)
                    self.node_cnt += 1
                else:
                    sz=len(leaf.parents)
                    phi_decrease= leaf.phi/sz - leaf.phi/(sz+1.0)
                    #delta_decrease=leaf[1]/sz - leaf[1]/(sz+1.0)
                    leaf.phi, leaf.delta=leaf.phi/(sz+1), leaf.delta/(sz+1)
                    self.update_ancestors(phi_decrease, leaf.delta,leaf.parents)
                    leaf.parents.append(self.mWorkHash)

    def update_ancestors(self, phi_decrease, delta, parents):
        return
        que=Queue.Queue()
        ldelta=delta
        for p in parents:
            que.put(p)
        while not que.empty():
            pcode=que.get()
            pnode=self.tt_lookup(pcode)
            if pnode == False:
                continue
            pnode.delta -= phi_decrease
            if pnode.phi > ldelta:
                phi_decrease=pnode.phi - delta
                pnode.phi=delta
                #for item in ps:
                #    que.put(item)
            ldelta=pnode.delta


    def tt_write(self, n):
        self.mTT[n.code]=n

    def tt_lookup(self, code):
        if code in self.mTT.keys():
            return self.mTT[code]
        else:
            #no this node
            return False

    def selectChild(self):
        delta1=INF
        delta2=INF
        best_child_node=None
        best_move=None
        ps=None
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                tcode=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                tnode=self.tt_lookup(tcode)
                assert(tnode)
                phi,delta=tnode.phi, tnode.delta
                if delta < delta1:
                    best_move=i
                    best_child_node=tnode
                    delta2=delta1
                    delta1=delta
                elif delta < delta2:
                    delta2 = delta
                if phi >= INF:
                    return tnode

        return best_child_node, delta2, best_move

    def phiSum(self):
        s=0
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                tcode=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                tnode=self.tt_lookup(tcode)
                s+=tnode.phi
                assert (tnode)
        return s

    def deltaMin(self):
        min_delta=INF
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                tcode = self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                tnode = self.tt_lookup(tcode)
                #print("pair ",pair)
                assert(tnode)
                min_delta=min(min_delta, tnode.delta)
        return min_delta

if __name__ == "__main__":
    pns=FPNS()
    state=[]
    print("FPNS")
    pns.fdfpns(state=state, toplay=HexColor.BLACK)
