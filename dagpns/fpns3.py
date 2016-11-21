from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from utils.unionfind import *
from zobrist.zobrist import *
from dagpns.commons import *
import Queue
import copy
import time

INF = 2000000000.0
EPSILON=1e-5

#when expanding a node, only update the MPN ancestors.
class FPNS:

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
        self.node_cnt=0
        self.mTT={}
        rootHash=self.zsh.get_hash(rootstate)
        rootNode=FNode(phi=1, delta=1, isexpanded=False, parents=[])
        self.tt_write(rootHash, rootNode)
        ite=0
        while(rootNode.phi > EPSILON and rootNode.delta >EPSILON):
            self.mHash = rootHash
            self.mState = copy.deepcopy(rootstate)
            self.mToplay = toplay
            self.mNode = self.tt_lookup(self.mHash)
            print("iteration ", ite, "root:", self.mHash, "root.phi, root.delta", rootNode.phi, rootNode.delta)
            self.selection()
            self.expansion()
            print("after expansion  \n")
            self.update_ancesotrs()
            ite +=1
            #if ite>3: break
            rootNode=self.tt_lookup(rootHash)
        print(rootNode.phi, rootNode.delta)
        print("1-Black, 2-White")
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
            assert(k>0)
            if node.phi+EPSILON>INF: k = 1
            assert(node)
            s += node.phi/k
        if s>INF: return INF
        return s

    def deltaMin(self, n):
        min_delta=INF
        for child_code,move in n.children:
            node = self.tt_lookup(child_code)
            assert (node)
            k=len(node.parents)
            assert(k>0)
            if node.delta+EPSILON>INF: k=1
            min_delta=min(min_delta, node.delta/k)
        return min_delta

    def selection(self):

        while self.mNode.isexpanded:
            #print(self.mState, self.mHash, self.mNode.phi, self.mNode.delta, self.mNode.children)
            best_move=-1
            best_val=INF
            best_child=None
            best_child_code=None
            for child_code,move in self.mNode.children:
                node=self.tt_lookup(child_code)
                assert(node)
                e_delta=node.delta
                if e_delta + EPSILON <INF and len(node.parents)!=0:
                    e_delta = e_delta/len(node.parents)
                if e_delta < best_val:
                    best_val = e_delta
                    best_move = move
                    best_child=node
                    best_child_code=child_code
            assert(best_move>-1)
            self.mState.append(best_move)
            self.mHash=best_child_code
            self.mToplay = HexColor.EMPTY - self.mToplay
            self.mNode=best_child
        toplay2 = HexColor.BLACK if len(self.mState)%2==0 else HexColor.WHITE
        assert (toplay2 == self.mToplay)
            #print("best-child", best_child_code)

    def expansion(self):
        self.mNode=self.tt_lookup(self.mHash)
        assert(self.mNode.isexpanded==False)
        self.mNode.asExpanded()
        avail_moves=[i for i in range(BOARD_SIZE**2) if i not in self.mState]
        self.mNode.children=[]
        existing=0
        toplay2 = HexColor.BLACK if len(self.mState)%2==0 else HexColor.WHITE
        assert (toplay2 == self.mToplay)
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
                toplay2=HexColor.BLACK if len(self.mState)%2==0 else HexColor.WHITE
                print("terminal:", self.mToplay, self.mState, "m=",m)
                assert(toplay2 == self.mToplay)
                phi,delta=(0,INF) if res==self.mToplay else (INF, 0)
                leaf=FNode(phi=phi, delta=delta, isexpanded=True)
                leaf.parents=[self.mHash]
                self.tt_write(child_code, leaf)
                self.node_cnt +=1
                if delta == 0:
                    print("impossible?")
                    self.mState.remove(m)
                    self.mToplay = HexColor.EMPTY - self.mToplay
                    break
            else:
                phi,delta=(1,1)
                leaf=FNode(phi=phi, delta=delta, isexpanded=False)
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
        if 0 and len(self.mExisting)>0:
            for c_code in self.mExisting:
                c_node=self.tt_lookup(c_code)
                assert(c_node)
                for p_code in c_node.parents:
                    if p_code not in Q.queue:
                        Q.put(p_code)
        while not Q.empty():
            code=Q.get()
            node=self.tt_lookup(code)
            oldphi, olddelta=node.phi, node.delta
            p,d=self.deltaMin(node), self.phiSum(node)
            if abs(p -oldphi) > EPSILON or abs(d-olddelta) >EPSILON:
                node.phi, node.delta=p,d
                self.tt_write(code, node)
                for p_code in node.parents:
                    if p_code not in Q.queue:
                        Q.put(p_code)

if __name__ == "__main__":


    for i in range(BOARD_SIZE ** 2):
        if i==3: continue
        pns2 = FPNS()
        print("openning ", i)
        start = time.time()
        state = [3,i]
        toplay = HexColor.BLACK if len(state) % 2 == 0 else HexColor.WHITE
        pns2.pns(state, toplay)
        end = time.time()
        print("solving opening", i, " time:", (end - start), "\n")

    for i in range(0*BOARD_SIZE ** 2):
        pns2 = FPNS()
        print("openning ", i)
        start=time.time()
        state = [i]
        toplay = HexColor.BLACK if len(state) % 2 == 0 else HexColor.WHITE
        pns2.pns(state, HexColor.WHITE)
        end=time.time()
        print("solving opening",i," time:",(end-start),"\n")

