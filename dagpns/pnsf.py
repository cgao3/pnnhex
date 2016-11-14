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
    def __init__(self, phi, delta, isexpanded, parents=None, children=None):
        self.phi=phi
        self.delta=delta
        self.isexpanded=isexpanded
        self.parents=parents
        self.children=children

    def setChildren(self, children_nodes):
        self.children=children_nodes

    def addParent(self, parent):
        if self.parents==None:
            self.parents=[parent]
        else:
            self.parents.append(parent)



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

        while(root.phi>EPSILON and root.delta >EPSILON):
            self.selection()
            self.expand()
            self.update_ancesotrs()

    def setProofDisproofNumber(self, n):
        if n.isexpanded:
            n.phi = INF
            n.delta = 0
            for c in n.children:
                n.phi=min(n.phi, c.delta)
                n.delta += c.phi
        else:
            val=self.evaluate(self.m)
            if(val==self.mToplay):
                n.phi, n.delta=(0,INF)
            elif(val==HexColor.EMPTY-self.mToplay):
                n.phi, n.delta=(INF,0)
            else:
                n.phi, n.delta=(1,1)

    def selection(self, n):
        val=INF
        best_node=None
        while n.isexpanded:
            best_node=None
            val=INF
            for c in n.children:
                if val < c.phi:
                    val=c.phi
                    best_node=c
            assert(best_node)
            n=best_node
        return n

    def expand(self):
        pass

    def update_ancesotrs(self):
        pass
