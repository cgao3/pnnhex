from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from utils.unionfind import *
from zobrist.zobrist import *
from .pns import winner,updateUF

import copy

INF = 2000000000.0
BOARD_SIZE=3

NORTH_EDGE=-1
SOUTH_EDGE=-2
WEST_EDGE=-3
EAST_EDGE=-4

EPSILON=1e-5


class PNSNode:
    def __init__(self, phi, delta, isexpanded, parents=None):
        self.phi=phi
        self.delta=delta
        self.isexpanded=isexpanded
        self.parents=parents


class PNSF:

    def __init__(self):
        self.mToplay=None
        self.mWorkstate=None
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

    def pns(self, rootstate, toplay):
        val=self.evaluate(rootstate)
        if val!=HexColor.EMPTY:
            print("Winner ", val)
            return
        self.mWorkHash=self.mWorkHash=self.zsh.get_hash(rootstate)
        root=PNSNode(phi=0, delta=0, isexpanded=False, parents=[])
        self.mToplay=toplay
        if val==toplay:
            root.phi, root.delta=(0,INF)
        elif val==HexColor.EMPTY-toplay:
            root.phi, root.delta=(INF,0)
        else:
            root.phi, root.delta=(1,1)
        self.node_cnt=0
        self.mTT={}
        self.mWorkstate=rootstate
        root=PNSNode(phi=1, delta=1, isexpanded=False)
        self.mCurrentNode=copy.copy(root)
        self.mAvailableMoves=[i for i in range(BOARD_SIZE**2) if i not in self.mWorkstate]
        self.mLeafsTrans=[]
        while(root.phi > EPSILON and root.delta >EPSILON):
            self.selection()
            self.expand()
            self.mLeafsTrans=[]
            self.update_ancesotrs()

    def tt_lookup(self, code):
        if code in self.mTT.keys():
            return self.mTT[code]
        return None

    def tt_write(self, code, n):
        self.mTT[code]=n

    def phiSum(self):
        s=0
        for i in self.mAvailableMoves:
            child_code=self.zsh.update_hash(self.mWorkHash, i, self.mToplay)
            node=self.tt_lookup(child_code)
            assert(node)
            s += node.phi/len(node.parents)
        return s

    def deltaMin(self):
        min_delta=INF
        for i in self.mAvailableMoves:
            child_code = self.zsh.update_hash(self.mWorkHash, i, self.mToplay)
            node = self.tt_lookup(child_code)
            assert (node)
            min_delta=min(min_delta, node.delta/len(node.parents))
        return min_delta

    def selection(self):

        while self.mCurrentNode.isexpanded:
            best_move=-1
            best_val=INF
            best_child=None
            for i in self.mAvailableMoves:
                code=self.zsh.update_hash(self.mWorkHash, i, self.mToplay)
                node=self.tt_lookup(code)
                assert(node)
                if node.delta > best_val:
                    best_val = node.delta
                    best_move = i
                    best_child=node
            assert(best_move>-1)
            self.mWorkstate.append(best_move)
            self.mWorkHash=self.zsh.update_hash(self.mWorkHash, best_move, self.mToplay)
            self.mToplay = HexColor.EMPTY - self.mToplay
            self.mAvailableMoves.remove(best_move)
            self.mCurrentNode = best_child


    def expand(self):
        assert(self.mCurrentNode.isexpanded==False)
        for i in self.mAvailableMoves:
            code=self.zsh.update_hash(code=self.mWorkHash, intmove=i, intplayer=self.mToplay)
            #check if the leaf has been there
            n=self.tt_lookup(code)
            if n:
                n.parents.append(i)
                self.tt_write(code, n)
                self.mLeafsTrans.append(code)
                continue
            assert(n==None)
            self.mWorkstate.append(i)
            self.mToplay = HexColor.EMPTY - self.mToplay
            res=self.evaluate(self.mWorkstate)
            if res!=HexColor.EMPTY:
                phi,delta=(0,INF) if res==self.mToplay else (INF, 0)
                leaf=PNSNode(phi=phi, delta=delta, isexpanded=True)
                leaf.parents=[i]
                self.tt_write(code, leaf)
                self.node_cnt +=1
                if delta == 0: break
            else:
                phi,delta=(1,1)
                leaf=PNSNode(phi=phi, delta=delta, isexpanded=False)
                leaf.parents=[i]
                self.tt_write(code, leaf)
                self.node_cnt +=1
            self.mWorkstate.remove(i)
            self.mToplay = HexColor.EMPTY - self.mToplay

    def update_ancesotrs(self):
        if len(self.mLeafsTrans)==0:
            pass
        else:
            pass
        while True:
            oldPhi, oldDelta=self.mCurrentNode.phi, self.mCurrentNode.delta
            self.phiSum()
            self.deltaMin()

        pass
