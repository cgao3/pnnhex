from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from zobrist.zobrist import *
from utils.unionfind import *
from dagpns.commons import *
import time

import copy

INF=200000000.0
EPSILON=1e-5

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
        root = Node(phi=INF, delta=INF, children=None)
        self.tt_write(self.mHash, root)
        self.mNode=root
        print("root state:", state)
        self.MID(root)
        print("root.phi, delta", root.phi, root.delta)
        print("1 for Black, 2 for White")
        if(root.phi<EPSILON):
            print(toplay, " Win")
        elif root.delta<EPSILON:
            print(HexColor.EMPTY-toplay, "Win")
        else:
            print("Unknown, something wrong?")

        print("number nodes expanded: ", self.node_cnt, "MID calls:", self.mid_calls)

    def MID(self, n):
        print("MID call: ", self.mid_calls, "state: ", self.mState, "toplay", self.mToplay, "node_cnt:", self.node_cnt)
        print(n.phi,n.delta)
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
            print("terminal")
            return

        print("before generating", n.children)
        self.generate_moves(n)
        print("after generating", n.children)
        k=n.nparent
        if k==0 or n.phi+EPSILON >INF or n.delta+EPSILON>INF: k=1
        phi_thre=self.deltaMin(n)
        delta_thre=self.phiSum(n)
        print("phi_thre", phi_thre)
        print("delta_thre", delta_thre)
        print("n.phi,n.delta",n.phi,n.delta)
        while n.phi > phi_thre and n.delta > delta_thre:
            print("n.phi,n.delta", n.phi, n.delta)
            best_child_node, delta2, best_move=self.selectChild(n)
            print("best_child_node", best_child_node)
            k=best_child_node.nparent
            if best_child_node.phi + EPSILON > INF or best_child_node.delta +EPSILON >INF:
                k=1
            assert(k>=1)
            best_child_node.phi = (n.delta - delta_thre)/k + best_child_node.phi
            e_delta=best_child_node.delta/k
            assert(n.phi > e_delta)
            print("e_delta", e_delta, "delta2", delta2, "n.phi", n.phi)
            assert(delta2 >= delta2)
            olddelta=best_child_node.delta
            best_child_node.delta=min(INF, (n.phi-e_delta)*k+olddelta, olddelta+(delta2-e_delta)*k+1)
            assert(best_move!=None)
            self.mState.append(best_move)
            self.mHash=self.zhash.update_hash(code=self.mHash, intmove=best_move, intplayer=self.mToplay)
            self.mToplay= HexColor.EMPTY - self.mToplay
            self.MID(best_child_node)
            self.mState.remove(best_move)
            self.mToplay = HexColor.EMPTY - self.mToplay
            self.mHash = self.zhash.update_hash(code=self.mHash, intmove=best_move, intplayer=self.mToplay)
            k=n.nparent
            if k==0 or n.phi+EPSILON >INF or n.delta+EPSILON>INF: k=1
            phi_thre=self.deltaMin(n)
            delta_thre=self.phiSum(n)
        n.phi=self.deltaMin(n)
        n.delta=self.phiSum(n)
        self.tt_write(self.mHash, n)
        print("write...")



    #write new positions to TT
    def generate_moves(self, n):
        if n.children!=None: return
        assert(n.children==None)
        n.children=[]
        for move in range(BOARD_SIZE**2):
            if move not in self.mState:
                child_code=self.zhash.update_hash(code=self.mHash, intmove=move, intplayer=self.mToplay)
                n.children.append((child_code,move))
                node=self.tt_lookup(child_code)
                if not node:
                    newnode=Node(phi=1, delta=1, nparent=1, children=None)
                    self.tt_write(child_code, newnode)
                    self.node_cnt += 1
                else:
                    node.nparent +=1
                    self.tt_write(child_code,node)
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
        best_e_delta=INF
        e_delta2=INF
        best_child_node=None
        best_move=None
        for child_code,move in n.children: 
            node=self.tt_lookup(child_code)
            assert(node) 
            k=node.nparent
            if node.phi +EPSILON >INF or node.delta+EPSILON>INF:
                k=1
            ephi, edelta=node.phi/k,node.delta/k
            print("k=",k, "phi,delta",ephi,edelta, "move",move)
            if edelta < best_e_delta: 
                best_child_node=node
                e_delta2=best_e_delta
                best_e_delta=edelta
                best_move=move
            elif edelta < e_delta2:
                e_delta2 = edelta
            if node.phi >= INF:
                return node, e_delta2, move

        return best_child_node, e_delta2, best_move

    def phiSum(self, n):
        s=0
        for child_code,move in n.children:
            node=self.tt_lookup(child_code)
            assert(node)
            k=node.nparent
            if node.phi + EPSILON >INF:
                k=1
            s+=node.phi/k
        return s

    def deltaMin(self, n):
        min_delta=INF
        for child_code,move in n.children:
            node = self.tt_lookup(child_code)
            assert(node)
            k=node.nparent
            if node.delta + EPSILON >INF:
                k=1
            min_delta=min(min_delta, node.delta/k)
        return min_delta

if __name__ == "__main__":

     pns2=DFPN()
     for i in range(1,BOARD_SIZE**2):
         if i==3: continue
         start=time.time()
         print("\nopenning ", i)
         state=[2]
         toplay=HexColor.BLACK if len(state)%2==0 else HexColor.WHITE
         pns2.dfpns(state=state, toplay=toplay)
         end=time.time()
         print("time cost solving opening",i," :",(end-start))
         break
