from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import  re
import os
from zobrist.zobrist import *
import argparse

#Smart Game Format
''''
First need to convert a .sgf game into a list of positions.
A directory has many .sgf games, so do them one by one.
Save the positions in a database (a text file),
and then remove the duplicates, save to a new file.
'''
class SgfUtil:

    pattern=r';[B|W]\[[a-zA-Z][1-9]+\]'
    result_pattern=r';[B|W]\[resign\]'
    result_pattern2=r'RE\[[B|W]\+\]'

    #convert a sgf Hex game to a list of positions, last
    #offset=k means positions at least have k move, k>=0, last move. for openning play sgfs, k=1 (discard empty board state)
    #for state-value data, offset is number of stones in the state, k must >=1.
    def __init__(self, boardsize, srcDir, outputname, offset=1, withvalue=False):
        self.boardsize=boardsize
        self.offset=offset
        self.withValue=withvalue
        self.outfilename=outputname
        self.sgfDir=srcDir

    def toStateActions(self, sgfgame, ret):
        game=re.findall(self.pattern, sgfgame)
        x=""
        for i, rawMove in enumerate(game):
            x +=rawMove[1:]+" "
            if i < self.offset:
                continue
            ret.append(x.strip())
        return ret

    def toStates(self, sgfgame, ret):
        game=re.findall(self.pattern, sgfgame)
        x=""
        for i, rawMove in enumerate(game):
            x +=rawMove[1:]+" "
            if i < self.offset-1:
                continue
            ret.append(x.strip())
        return ret

    def toStateValues(self, sgfgame, ret):
        assert(self.offset>=1)
        result_str=re.findall(self.result_pattern2, sgfgame)
        self.toStates(sgfgame, ret)
        result_str=result_str[0]
        winner=HexColor.BLACK if result_str[3]=='B' else HexColor.WHITE
        for i, state in enumerate(ret):
            #print(state, len(state)%2)
            if((len(state.split())%2) + 1==winner):
                state = state + " " + "1"
            else:
                state = state + " " + "0"
            ret[i]=state

    def writeToFile(self, str_positions):
        out=open(self.outfilename, "a")
        for posi in str_positions:
            out.write(posi)
            out.write('\n')
        out.close()

    #convert all .sgf games in a directory
    def performConvert(self):
        onlyfiles = [f for f in os.listdir(self.sgfDir) if os.path.isfile(os.path.join(self.sgfDir, f))]
        print("processing convert...")
        for f in onlyfiles:
            print("converting", f)
            infile=open(os.path.join(self.sgfDir, f), "r")
            sgfgame=infile.read()
            data=[]
            if not self.withValue:
                self.toStateActions(sgfgame, data)
            else: #state-value
                self.toStateValues(sgfgame,data)
            self.writeToFile(data)
            infile.close()
        print("removing duplicates...")
        self.removeDuplicatesAndWrite(self.outfilename)
        print("Done.")

    #remove duplicate and save in a new file
    def removeDuplicatesAndWrite(self, file_name):
        tt={}
        hashUtil=ZobristHash(boardsize=self.boardsize)
        with open(file_name) as f:
            for line in f:
                line=line.strip()
                movesquence=line.split()
                if self.withValue:
                    movesquence=movesquence[:-1]
                intmoveseq=[]
                for m in movesquence:
                    move=m[2:4]
                    x=ord(move[0].lower())-ord('a')
                    y=ord(move[1])-ord('0')-1
                    intmoveseq.append(x*self.boardsize+y)
                code=hashUtil.get_hash(intmoveseq)
                tt[code]=line

        outfile=file_name+"_no_duplicates"
        print("size: ", len(tt.values()))
        print("saved as", outfile)
        with open(outfile, "w+") as f:
            for line in tt.values():
                #print(line)
                f.write(line)
                f.write('\n')
def test():
    sgf = ""
    with open("/home/cgao/Documents/hex_data/8x8/0000.sgf") as f:
        sgf = f.read()
    print(sgf)
    data = []
    sgfutil=SgfUtil(boardsize=8, srcDir=None, outputname=None)
    sgfutil.toStateValues(sgf, data)
    print(data)

def process0():
    sgfutil=SgfUtil(boardsize=8, offset=1,
                    srcDir="/home/cgao/Documents/hex_data/8x8",
                    outputname="8x8-raw0.txt",
                    withvalue=True)
    sgfutil.performConvert()

def process1():
    sgfutil=SgfUtil(boardsize=8, offset=1,
                    srcDir="/home/cgao/Documents/hex_data/8x8-1-1/2/8x8-mohex-vs-mohex-1-stone-opening",
                    outputname="8x8-raw21.txt",
                    withvalue=True)
    sgfutil.performConvert()

def process2():
    sgfutil=SgfUtil(boardsize=8, offset=2,
                    srcDir="/home/cgao/Documents/hex_data/8x8-two-stones-openings/silurian_data/8x8-mohex-mohex-14s-vs-wolve-wolve-12s",
                    outputname="8x8-SA23.txt",
                    withvalue=True)
    sgfutil.performConvert()

def processRemoveDuplicates():
    filename="train.txt"
    sgfutil=SgfUtil(boardsize=8, offset=2,
                    srcDir=None, outputname=None, withvalue=True)
    sgfutil.removeDuplicatesAndWrite(filename)

if __name__ == "__main__":
    #process2()
    processRemoveDuplicates()
