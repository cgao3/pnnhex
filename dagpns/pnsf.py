from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from utils.unionfind import *
from zobrist.zobrist import *
from dagpns.pns import winner,updateUF

import Queue
import copy

INF = 2000000000.0
BOARD_SIZE=4

NORTH_EDGE=-1
SOUTH_EDGE=-2
WEST_EDGE=-3
EAST_EDGE=-4

EPSILON=1e-5


class PNSNode:
    def __init__(self, phi, delta, isexpanded, parents=None, children=None):
        self.phi=phi
        self.delta=delta
        self.isexpanded=isexpanded
        self.parents=parents
        self.children=children

    def asExpanded(self):
        self.isexpanded=True

class PNSF:

    def __init__(self):
        self.mToplay=None
        self.mState=None
        self.zsh=ZobristHash(BOARD_SIZE)

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
            return outcome
        for m in range(BOARD_SIZE**2):
            if m not in moveseq:
                b,w=copy.deepcopy(blackUF), copy.deepcopy(whiteUF)
                moveseq.append(m)
                b,w=updateUF(moveseq, b,w,m,self.mToplay)
                res=winner(b,w)
                if res!=HexColor.EMPTY:
                    assert(res==self.mToplay)
                    moveseq.remove(m)
                    return res
                moveseq.remove(m)
        return HexColor.EMPTY

    def pns(self, rootstate, toplay):
        self.mToplay=toplay
        val=self.evaluate(rootstate)
        if val!=HexColor.EMPTY:
            print("Winner ", val)
            return
        self.node_cnt=0
        self.mTT={}
        rootHash=self.zsh.get_hash(rootstate)
        self.mState=rootstate
        rootNode=PNSNode(phi=1, delta=1, isexpanded=False, parents=[])
        self.tt_write(rootHash, rootNode)
        ite=0
        while(rootNode.phi > EPSILON and rootNode.delta >EPSILON):
            self.mHash = rootHash
            self.mState = copy.deepcopy(rootstate)
            self.mToplay = toplay
            self.mNode = self.tt_lookup(self.mHash)
            print("iteration ", ite, "root.phi, root.delta", rootNode.phi, rootNode.delta)
            print("root ", self.mHash)
            self.selection()
            print("selected: ", self.mHash)
            self.expansion()
            print("after expansion  \n")
            self.update_ancesotrs()
            ite +=1
            #if ite>3: break
            rootNode=self.tt_lookup(rootHash)
        print(rootNode.phi, rootNode.delta)
        print("number of nodes expanded:", self.node_cnt)
        if rootNode.phi <EPSILON:
            print(toplay, "wins")
        else:
            print(toplay, "loses")

    def tt_lookup(self, code):
        if code in self.mTT.keys():
            return self.mTT[code]
        return False

    def tt_write(self, code, n):
        self.mTT[code]=n

    def phiSum(self, n):
        s=0
        for child_code,move in n.children:
            node=self.tt_lookup(child_code)
            k = len(node.parents)
            if k == 0 or node.phi+EPSILON>INF: k = 1
            assert(node)
            s += node.phi/k
        return s

    def deltaMin(self, n):
        min_delta=INF
        for child_code,move in n.children:
            node = self.tt_lookup(child_code)
            assert (node)
            k=len(node.parents)
            if k==0 or node.delta+EPSILON>INF: k=1
            min_delta=min(min_delta, node.delta/k)
        return min_delta

    def selection(self):
        print("start selection..")
        ite=0
        while self.mNode.isexpanded:
            #print(self.mState, self.mHash, self.mNode.phi, self.mNode.delta, self.mNode.children)
            best_move=-1
            best_val=INF
            best_child=None
            best_child_code=None
            for child_code,move in self.mNode.children:
                node=self.tt_lookup(child_code)
                assert(node)
                if node.delta < best_val:
                    best_val = node.delta
                    best_move = move
                    best_child=node
                    best_child_code=child_code
            assert(best_move>-1)
            self.mState.append(best_move)
            self.mHash=best_child_code
            self.mToplay = HexColor.EMPTY - self.mToplay
            self.mNode=best_child

            #print("best-child", best_child_code)
        print("end seelction")
    def expansion(self):
        self.mNode=self.tt_lookup(self.mHash)
        assert(self.mNode.isexpanded==False)
        self.mNode.asExpanded()
        avail_moves=[i for i in range(BOARD_SIZE**2) if i not in self.mState]
        self.mNode.children=[]
        existing=0
        self.mExisting=[]
        for m in avail_moves:
            child_code=self.zsh.update_hash(code=self.mHash, intmove=m, intplayer=self.mToplay)
            self.mNode.children.append((child_code,m))
            #check if the leaf has been there
            node=self.tt_lookup(child_code)
            if node:
                node.parents.append(self.mHash)
                self.tt_write(child_code, node)
                self.mExisting.append(child_code)
                existing +=1
                continue
            assert(node==False)
            self.mState.append(m)
            self.mToplay = HexColor.EMPTY - self.mToplay
            res=self.evaluate(self.mState)
            if res!=HexColor.EMPTY:
                phi,delta=(0,INF) if res==self.mToplay else (INF, 0)
                leaf=PNSNode(phi=phi, delta=delta, isexpanded=True)
                leaf.parents=[self.mHash]
                self.tt_write(child_code, leaf)
                self.node_cnt +=1
                if delta == 0: break
            else:
                phi,delta=(1,1)
                leaf=PNSNode(phi=phi, delta=delta, isexpanded=False)
                leaf.parents=[self.mHash]
                self.tt_write(child_code, leaf)
                self.node_cnt +=1
            self.mState.remove(m)
            self.mToplay = HexColor.EMPTY - self.mToplay
        self.tt_write(self.mHash, self.mNode)
        print("expanding ", self.mHash, "has", len(self.mNode.children), "nodes, ", existing, "existing")

    def update_ancesotrs(self):
        Q = Queue.Queue()
        Q.put(self.mHash)
        if len(self.mExisting)>0:
            for c_code in self.mExisting:
                c_node=self.tt_lookup(c_code)
                assert(c_node)
                for p_code in c_node.parents:
                    if p_code not in Q.queue:
                        Q.put(p_code)
        while not Q.empty():
            code=Q.get()
            self.mNode=self.tt_lookup(code)
            oldphi, olddelta=self.mNode.phi, self.mNode.delta
            p,d=self.deltaMin(self.mNode), self.phiSum(self.mNode)
            if abs(p -oldphi) > EPSILON or abs(d-olddelta) >EPSILON:
                self.mNode.phi, self.mNode.delta=p,d
                self.tt_write(code, self.mNode)
                for p_code in self.mNode.parents:
                    if p_code not in Q.queue: Q.put(p_code)

if __name__ == "__main__":
    pnsf=PNSF()
    s=[]
    toplay=HexColor.BLACK if len(s)%2==0 else HexColor.WHITE
    pnsf.pns(s, toplay)
    for i in range(BOARD_SIZE ** 2):
        pns2 = PNSF()
        print("openning ", i)
        state = [i]
        pns2.pns(state, HexColor.WHITE)
