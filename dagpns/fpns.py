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
        root = Node(phi=INF, delta=INF, code=self.mWorkHash, parents=[])
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
                (n.phi, n.delta)=(0, INF)
            else:
                (n.phi, n.delta)=(INF, 0)
            print("Terminal state: ", self.mWorkState)

            return

        self.generate_moves()
        while n.phi > self.deltaMin() and n.delta > self.phiSum():
            c_best, delta2, best_move=self.selectChild()
            c_best.phi = n.delta - self.phiSum() + c_best.phi/len(c_best.parents)
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
                    #leaf.phi, leaf.delta=leaf.phi/(sz+1), leaf.delta/(sz+1)
                    #self.update_ancestors(leaf, phi_decrease,leaf.delta)
                    leaf.parents.append(self.mWorkHash)

    def update_ancestors(self, node, phi_decrease, child_delta):
        #return
        if node == False: return
        if phi_decrease > 0 or node.phi > child_delta:
            if phi_decrease >0:
                node.delta -= phi_decrease
            if node.phi > child_delta:
                phi_decrease = node.phi - child_delta
                node.phi = child_delta
            else:
                phi_decrease = 0
            for pnode_code in node.parents:
                pnode=self.tt_lookup(code=pnode_code)
                if pnode:
                    self.update_ancestors(pnode, phi_decrease, node.delta)
        else:
            return

    def tt_write(self, n):
        self.mTT[n.code]=n

    def tt_lookup(self, code):
        if code in self.mTT.keys():
            return self.mTT[code]
        else:
            #no this node
            return False

    def selectChild(self):
        delta1=INF+1
        delta2=INF+1
        best_child_node=None
        best_move=None
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                tcode=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                tnode=self.tt_lookup(tcode)
                assert(tnode)
                phi=INF if tnode.phi >=INF else tnode.phi/len(tnode.parents)
                delta=INF if tnode.delta >=INF else tnode.delta/len(tnode.parents)
                if delta < delta1:
                    best_move=i
                    best_child_node=tnode
                    delta2=delta1
                    delta1=delta
                elif delta < delta2:
                    delta2 = delta
                if phi >= INF:
                    return (tnode, delta2, best_move)
        print(best_child_node, delta2, best_move)
        assert(best_child_node)
        assert(delta2)
        assert(best_move!=None)
        return best_child_node, delta2, best_move

    def phiSum(self):
        s=0
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                tcode=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                tnode=self.tt_lookup(tcode)
                s+=tnode.phi/len(tnode.parents)
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
                min_delta=min(min_delta, tnode.delta/len(tnode.parents))
        return min_delta

if __name__ == "__main__":
    pns=FPNS()
    state=[]
    print("FPNS")
    pns.fdfpns(state=state, toplay=HexColor.BLACK)
