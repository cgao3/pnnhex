from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import  re
import os
from zobrist.zobrist import *
import argparse

class SGFPositionActionUtil:

    pattern=r';[B|W]\[[a-zA-Z][1-9]+\]'
    def __init__(self, srcdir, outfilename, offset=1):
        self.srcdir=srcdir
        self.outfilename=outfilename
        self.outfilewriter=open(outfilename, "w")
        self.offset=offset

    def strSGFtoPositions(self, strSGF, movesquence):
        game = re.findall(self.pattern, strSGF)
        x = ""
        for i, rawMove in enumerate(game):
            x += rawMove[1:] + " "
            if i < self.offset:
                continue
            movesquence.append(x.strip())
        return movesquence

    def doConvertInDir(self):
        onlyfiles = [f for f in os.listdir(self.srcdir) if os.path.isfile(os.path.join(self.srcdir, f))]
        print("processing convert...")
        for f in onlyfiles:
            print("converting", f)
            infile = open(os.path.join(self.srcdir, f), "r")
            strSGFGame = infile.read()
            positionActionsList = []
            self.strSGFtoPositions(strSGFGame, positionActionsList)
            self.writePositionActions(positionActionsList)
            infile.close()
        print("Done, position-actions writing to", self.outfilename)
        self.outfilewriter.close()

    def writePositionActions(self, positionList):
        for posi in positionList:
            self.outfilewriter.write(posi)
            self.outfilewriter.write('\n')
        self.outfilewriter.flush()

    @staticmethod
    def removeDuplicates(boardsize, infilename):
        tt = {}
        hashUtil = ZobristHash(boardsize=boardsize)
        with open(infilename) as f:
            for line in f:
                line = line.strip()
                movesquence = line.split()
                intmoveseq = []
                for m in movesquence:
                    move = m[2:4]
                    x = ord(move[0].lower()) - ord('a')
                    y = ord(move[1]) - ord('0') - 1
                    intmoveseq.append(x * boardsize + y)
                code = hashUtil.get_hash(intmoveseq)
                tt[code] = line

        outfile = infilename + "_no_duplicates"
        print("size: ", len(tt.values()))
        print("saved as", outfile)
        with open(outfile, "w") as f:
            for line in tt.values():
                # print(line)
                f.write(line)
                f.write('\n')

    @staticmethod
    def removeDataAppearsInMain(boardsize, filenameMain, filenameTest):
        ttTest = {}
        hashUtil = ZobristHash(boardsize=boardsize)
        with open(filenameTest) as f:
            for line in f:
                line = line.strip()
                movesquence = line.split()
                intmoveseq = []
                for m in movesquence:
                    move = m[2:4]
                    x = ord(move[0].lower()) - ord('a')
                    y = ord(move[1]) - ord('0') - 1
                    intmoveseq.append(x * boardsize + y)
                code = hashUtil.get_hash(intmoveseq)
                ttTest[code] = line

        count=0
        count2=0
        out=open(filenameMain+"_2", "w")
        with open(filenameMain) as f:
            for line in f:
                line = line.strip()
                movesquence = line.split()
                intmoveseq = []
                for m in movesquence:
                    move = m[2:4]
                    x = ord(move[0].lower()) - ord('a')
                    y = ord(move[1]) - ord('0') - 1
                    intmoveseq.append(x * boardsize + y)
                code = hashUtil.get_hash(intmoveseq)
                if not code in ttTest:
                    out.write(line)
                    out.write('\n')
                    count2 +=1
                count +=1
        out.close()
        print("unique position size in test:", len(ttTest), "pre-size", count, "now size: ", count2)
        with open(filenameTest+"_2","w") as f:
            for line in ttTest.values():
                f.write(line)
                f.write('\n')


class SGFPositionValueUtil(object):
    pattern=r';[B|W]\[[a-zA-Z][1-9]+\]'
    winnerPattern=r'RE\[[B|W]\+\]'

    def __init__(self, srcdir, outfilename, offset=1):
        self.srcdir=srcdir
        self.outfilename=outfilename
        self.offset=offset
        self.outwriter=open(self.outfilename,"w")

    def toPositions(self, strSGF, ret):
        game=re.findall(self.pattern, strSGF)
        x=""
        for i, rawMove in enumerate(game):
            x +=rawMove[1:]+" "
            if i < self.offset-1:
                continue
            ret.append(x.strip())
        return ret

    def toPositionValues(self, strSGF, positionValuesList):
        assert(self.offset>=1)
        resultStr=re.findall(self.winnerPattern, strSGF)
        self.toPositions(strSGF, positionValuesList)
        resultStr=resultStr[0]
        winner=HexColor.BLACK if resultStr[3]=='B' else HexColor.WHITE
        for i, posi in enumerate(positionValuesList):
            #print(state, len(state)%2)
            if((len(posi.split())%2) + 1==winner):
                posi = posi + " " + "1"
            else:
                posi = posi + " " + "0"
            positionValuesList[i]=posi

        return positionValuesList

    def writePositionValuesList(self, pvList):
        for posi in pvList:
            self.outwriter.write(posi)
            self.outwriter.write('\n')
        self.outwriter.flush()

    def doConvertInDir(self):
        onlyfiles = [f for f in os.listdir(self.srcdir) if os.path.isfile(os.path.join(self.srcdir, f))]
        print("processing convert...")
        for f in onlyfiles:
            print("converting", f)
            infile=open(os.path.join(self.srcdir, f), "r")
            sgfGame=infile.read()
            pvList=[]
            self.toPositionValues(sgfGame, pvList)
            self.writePositionValuesList(pvList)
            infile.close()
        print("Done, position-values saved in", self.outfilename)
        self.outwriter.close()


    @staticmethod
    def postprocess(boardsize, positionValuesFileName):
        print("position-value postprocessing")
        tt={}
        with open(positionValuesFileName) as f:
            for line in f:
                line=line.strip()
                movesquence=line.split()
                value = int(movesquence[-1])
                movesquence=movesquence[:-1]
                assert(value==0 or value==1)
                tenaryBoard=[0]*(boardsize*boardsize)
                turn=HexColor.BLACK
                for m in movesquence:
                    move=m[2:4]
                    x=ord(move[0].lower())-ord('a')
                    y=ord(move[1])-ord('0')-1
                    tenaryBoard[x*boardsize+y]=turn
                    turn = HexColor.EMPTY - turn
                code=''.join(map(str,tenaryBoard))
                if code in tt:
                    mq, one_count, zero_count=tt[code]
                    if value==1:
                        one_count +=1
                    else:
                        zero_count +=1
                    tt[code]=(mq, one_count, zero_count)
                else:
                    one_count=0
                    zero_count=0
                    if value==1:
                        one_count=1
                    else: zero_count=1
                    tt[code]=(movesquence, one_count, zero_count)

        outfile=positionValuesFileName+"-post"
        print("size: ", len(tt))
        print("saved as", outfile)
        with open(outfile, "w") as f:
            for line in tt.values():
                #print(line)
                mq, one_count, zero_count = line
                for m in mq:
                    f.write(m+' ')
                res=(one_count)*1.0/(one_count+zero_count)
                f.write(repr(res)+'\n')


def RewardAugment(srcPositionAction, srcPositionValue, outputname, boardsize=8):
    mydict={}
    print("reward augmenting..")
    with open(srcPositionAction) as fpa:
        for line in fpa:
            moveseq=line.split()
            tenaryBoard=[0]*(boardsize*boardsize)
            turn=HexColor.BLACK
            for m in moveseq:
                move=m[2:4]
                x = ord(move[0].lower()) - ord('a')
                y = ord(move[1]) - ord('0') - 1
                tenaryBoard[x * boardsize + y] = turn
                turn = HexColor.EMPTY - turn
            code = ''.join(map(str, tenaryBoard))
            mydict[code]=True

    fpv=open(srcPositionValue,"r")
    fout=open(outputname,"w")
    for line in fpv:
        line=line.strip()
        arr=line.split()
        S=arr[:-1]
        V=arr[-1]
        assert(-1-0.001<float(V)<1+0.001)
        tenaryBoard = [0] * (boardsize * boardsize)
        turn = HexColor.BLACK
        for m in S:
            move = m[2:4]
            x = ord(move[0].lower()) - ord('a')
            y = ord(move[1]) - ord('0') - 1
            tenaryBoard[x * boardsize + y] = turn
            turn = HexColor.EMPTY - turn
        code = ''.join(map(str, tenaryBoard))
        if(code in mydict):
            fout.write(line+'\n')
    fpv.close()
    fout.close()
    print("Done.")

def process0():
    outfilename="8x8-2.txt"
    putil=SGFPositionActionUtil(srcdir="/home/cgao3/Documents/hex_data/8x8-2stones-open", outfilename=outfilename, offset=2)
    putil.doConvertInDir()
    #putil.removeDuplicates(boardsize=8, infilename=outfilename)

def positionPostprocess():
    dataMain="8x8main.txt"
    dataPart="part.txt"
    SGFPositionActionUtil.removeDataAppearsInMain(boardsize=8, filenameMain=dataMain, filenameTest=dataPart)

def positionRemoveDuplicates():
    dataMain="pa.txt"
    SGFPositionActionUtil.removeDuplicates(boardsize=8, infilename=dataMain)

def process1():
    outfilename = "8x8-a4.txt"
    putil = SGFPositionActionUtil(srcdir="/home/cgao3/Documents/hex_data/a4", outfilename=outfilename,
                                  offset=1)
    putil.doConvertInDir()

def vprocess1():
    outfilename="8x8-v1.txt"
    vutil=SGFPositionValueUtil(srcdir="/home/cgao3/Documents/hex_data/8x8", outfilename=outfilename, offset=1)
    vutil.doConvertInDir()

def vprocess2():
    outfilename="8x8-v2.txt"
    vutil=SGFPositionValueUtil(srcdir="/home/cgao3/Documents/hex_data/8x8-2stones-open", outfilename=outfilename, offset=2)
    vutil.doConvertInDir()
def vprocessa4():
    outfile="8x8-v-a4.txt"
    vutil = SGFPositionValueUtil(srcdir="/home/cgao3/Documents/hex_data/a4", outfilename=outfile,
                                 offset=1)
    vutil.doConvertInDir()

def vpostprocess():
    SGFPositionValueUtil.postprocess(boardsize=8, positionValuesFileName="8x8-v-main.txt")

if __name__ == "__main__":
    #process0()
    #vpostprocess()
    #process1()
    #positionRemoveDuplicates()
    RewardAugment(srcPositionAction="storage/position-action/8x8/data.txt", srcPositionValue="storage/position-value/8x8/data.txt", outputname="rml-data.txt")