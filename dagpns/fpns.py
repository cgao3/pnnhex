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
BOARD_SIZE=3

NORTH_EDGE=-1
SOUTH_EDGE=-2
WEST_EDGE=-3
EAST_EDGE=-4

EPSILON=1e-5

class FPNS:
    def __init__(self):
        self.mWorkHash=0
        self.mToplay=None
        self.mWorkState=None
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
            print("possible?")
            return outcome
        for m in range(BOARD_SIZE**2):
            if m not in moveseq:
                b,w=copy.deepcopy(blackUF), copy.deepcopy(whiteUF)
                moveseq.append(m)
                b,w=updateUF(moveseq, b,w,m,self.mToplay)
                res=winner(b,w)
                moveseq.remove(m)
                if res!=HexColor.EMPTY:
                    return res
        return winner(blackUF, whiteUF)

    def fdfpns(self,state, toplay):
        self.node_cnt = 0
        self.mid_calls = 0
        self.mToplay = toplay
        self.mWorkState=state
        self.rootToplay=toplay
        self.mTT={}
        self.mWorkHash=self.zhash.get_hash(intstate=state)
        root = Node(phi=INF, delta=INF, code=self.mWorkHash, parents=[])
        self.MID(root)
        if(abs(root.phi) < EPSILON or root.delta >= INF-EPSILON):
            print(toplay, " Win")
        elif abs(root.delta) < EPSILON:
            print(toplay, "Lose")
        else:
            print("Unknown, something wrong?")

        print("number nodes expanded: ", self.node_cnt)
        print("number of MID calls, ", self.mid_calls)

    def MID(self, n):
        self.mid_calls +=1
        assert(len(self.mWorkState)%2+1==self.mToplay)
        print("MID calls, ", self.mid_calls, "num nodes ", self.node_cnt)
        outcome=self.evaluate(self.mWorkState)
        if (outcome!=HexColor.EMPTY):
            if outcome==self.mToplay:
                (n.phi, n.delta)=(0, INF)
            else:
                (n.phi, n.delta)=(INF, 0)
            print("Terminal state: ", self.mWorkState)

            return

        self.generate_moves()
        delta_thre=self.phiSum()
        phi_thre = self.deltaMin()
        while n.phi > phi_thre and n.delta > delta_thre:
            c_best, delta2, best_move=self.selectChild()
            c_best.phi = n.delta - delta_thre + c_best.phi/len(c_best.parents)
            c_best.delta=min(n.phi, (delta2*(1+0.25)))
            self.mWorkState.append(best_move)
            self.mWorkHash=self.zhash.update_hash(code=self.mWorkHash, intmove=best_move, intplayer=self.mToplay)
            self.mToplay= HexColor.EMPTY - self.mToplay
            self.MID(c_best)
            self.mWorkState.remove(best_move)
            self.mToplay = HexColor.EMPTY - self.mToplay
            self.mWorkHash = self.zhash.update_hash(code=self.mWorkHash, intmove=best_move, intplayer=self.mToplay)
            delta_thre=self.phiSum()
            phi_thre=self.deltaMin()
        n.phi=self.deltaMin()
        n.delta=self.phiSum()
        self.tt_write(n)

    #write new positions to TT
    def generate_moves(self):
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                child_code=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                leaf=self.tt_lookup(child_code)
                if not leaf:
                    n=Node(code=child_code, phi=1.0, delta=1.0, parents=[self.mWorkHash])
                    self.tt_write(n)
                    self.node_cnt += 1
                else:
                    sz=len(leaf.parents)
                    phi_decrease= leaf.phi/sz - leaf.phi/(sz+1.0)
                    #leaf.phi, leaf.delta=leaf.phi/(sz+1.0), leaf.delta/(sz+1.0)
                    for p in leaf.parents:
                        p_node=self.tt_lookup(p)
                        self.update_ancestors(p_node, phi_decrease,leaf.delta)
                    leaf.parents.append(self.mWorkHash)
                    self.tt_write(leaf)

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
            self.tt_write(node)
            for pnode_code in node.parents:
                pnode=self.tt_lookup(code=pnode_code)
                #if pnode:
                    #self.update_ancestors(pnode, phi_decrease, node.delta)
        else:
            return

    def tt_write(self, n):
        self.mTT[n.code]=n

    def tt_lookup(self, code):
        if code in self.mTT.keys():
            return self.mTT[code]
        else:
            return False

    def selectChild(self):
        delta_best=INF+EPSILON
        delta2_best=INF+EPSILON
        best_child_node=None
        best_move=None
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                child_code=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                child_node=self.tt_lookup(child_code)
                assert(child_node)
                #phi, delta=child_node.phi, child_node.delta
                phi=INF if child_node.phi >=INF-EPSILON else child_node.phi/len(child_node.parents)
                delta=INF if child_node.delta >=INF-EPSILON else child_node.delta/len(child_node.parents)
                print(phi, delta)
                if delta < delta_best:
                    best_move=i
                    best_child_node=child_node
                    delta2_best=delta_best
                    delta_best=delta
                elif delta < delta2_best:
                    delta2_best = delta
                if phi >= INF-EPSILON:
                    return (child_node, delta2_best, i)
        print("state ", self.mWorkState)
        print(best_child_node, delta2_best, best_move)
        assert(best_child_node)
        assert(delta2_best<=INF)
        assert(best_move!=None)
        return best_child_node, delta2_best, best_move

    def phiSum(self):
        s=0
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                child_code=self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                child_node=self.tt_lookup(child_code)
                phi = INF if child_node.phi > INF -EPSILON else child_node.phi/len(child_node.parents)
                s+= phi
                assert (child_node)
        return s

    def deltaMin(self):
        min_delta=INF
        for i in range(BOARD_SIZE**2):
            if i not in self.mWorkState:
                child_code = self.zhash.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
                child_node = self.tt_lookup(child_code)
                assert(child_node)
                delta = INF if child_node.delta > INF -EPSILON else child_node.delta/len(child_node.parents)
                min_delta=min(min_delta, delta)
        return min_delta

if __name__ == "__main__":
    pns=FPNS()
    state=[]
    print("FPNS")
    pns.fdfpns(state=state, toplay=HexColor.BLACK)
