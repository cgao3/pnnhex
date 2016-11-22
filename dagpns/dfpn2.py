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
        self.mHash=None
        self.mTT=None
        self.mToplay=None
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

    def dfpns(self, state, toplay):
        self.mToplay = toplay
        self.mState=state
        self.rootToplay=toplay
        self.mTT={}
        self.node_cnt=self.mid_calls=0
        self.mHash=self.zhash.get_hash(intstate=state)
        root = Node(phi=INF, delta=INF, code=self.mHash, children=None)
        self.tt_write(self.mHash, root)
        self.mNode=root
        print("root state:", state)
        self.MID(root)
        print("1 for Black, 2 for White")
        if(root.delta>=INF):
            print(toplay, " Win")
        elif root.delta == 0:
            print(HexColor.EMPTY-toplay, "Win")
        else:
            print("Unknown, something wrong?")

        print("number nodes expanded: ", self.node_cnt, "MID calls:", self.mid_calls)

    def MID(self, n):
        print("MID call: ", self.mid_calls, "state: ", self.mState, "toplay=", self.mToplay)
        assert(len(self.mState)%2+1==self.mToplay)
        self.mid_calls +=1
        outcome, is_terminal=self.evaluate(self.mState)
        if (outcome!=HexColor.EMPTY):
            if is_terminal == True:
                print("possible?")
                n.phi, n.delta = (0,INF) if outcome==self.mToplay else (INF, 0)
            if outcome==self.mToplay:
                (n.phi, n.delta)=(0, INF)
            else:
                (n.phi, n.delta)=(INF, 0)
            self.tt_write(self.mHash, n)
            return

        n=self.generate_moves(n)
        print("after generating", n.children)
        phi_thre=self.deltaMin(n)
        delta_thre=self.phiSum(n)
        while n.phi > phi_thre and n.delta > delta_thre:
            best_child_node, delta2, best_move=self.selectChild(n)
            best_child_node.phi = n.delta - delta_thre + best_child_node.phi
            best_child_node.delta=min(n.phi, delta2+1)
            assert(best_move!=None)
            self.mState.append(best_move)
            self.mHash=self.zhash.update_hash(code=self.mHash, intmove=best_move, intplayer=self.mToplay)
            self.mToplay= HexColor.EMPTY - self.mToplay
            self.MID(best_child_node)
            self.mState.remove(best_move)
            self.mToplay = HexColor.EMPTY - self.mToplay
            self.mHash = self.zhash.update_hash(code=self.mHash, intmove=best_move, intplayer=self.mToplay)
            phi_thre=self.deltaMin(n)
            delta_thre=self.phiSum(n)
        n.phi=self.deltaMin(n)
        n.delta=self.phiSum(n)
        self.tt_write(self.mHash, n)

    #write new positions to TT
    def generate_moves(self, n):
        assert(n.children==None)
        n.children=[]
        for move in range(BOARD_SIZE**2):
            if move not in self.mState:
                child_code=self.zhash.update_hash(code=self.mHash, intmove=move, intplayer=self.mToplay)
                n.children.append((child_code,move))
                node=self.tt_lookup(child_code)
                if not node:
                    newnode=Node(phi=1, delta=1)
                    self.tt_write(child_code, newnode)
                    self.node_cnt += 1
        self.tt_write(self.mHash, n)
        return n
    def tt_write(self, code, n):
        self.mTT[code]=n

    def tt_lookup(self, code):
        if code in self.mTT.keys():
            return self.mTT[code]
        else:
            return False

    def selectChild(self, n):
        best_delta=INF
        delta2=INF
        best_phi=INF
        best_child_code=None
        best_child_node=None
        best_move=None
        for child_code,move in n.children: 
            node=self.tt_lookup(child_code)
            assert(node) 
            phi, delta=node.phi,node.delta
            if delta < best_delta: 
                best_child_node=node
                delta2=best_delta
                best_delta=delta
                best_phi=phi
                best_move=move
            elif delta < delta2:
                delta2 = delta
            if phi >= INF:
                return node, delta2, move

        return best_child_node, delta2, move

    def phiSum(self, n):
        s=0
        for child_code,move in n.children:
            node=self.tt_lookup(child_code)
            assert(node)
            s+=node.phi
        return s

    def deltaMin(self, n):
        min_delta=INF
        for child_code,move in n.children:
            node = self.tt_lookup(child_code)
            assert(node)
            min_delta=min(min_delta, node.delta)
        return min_delta

if __name__ == "__main__":

     pns2=DFPN()
     for i in range(BOARD_SIZE**2):
         if i==3: continue
         start=time.time()
         print("\nopenning ", i)
         state=[3,i]
         toplay=HexColor.BLACK if len(state)%2==0 else HexColor.WHITE
         pns2.dfpns(state=state, toplay=toplay)
         end=time.time()
         print("time cost solving opening",i," :",(end-start))
         break
