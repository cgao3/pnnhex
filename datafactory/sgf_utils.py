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

    #convert a sgf Hex game to a list of positions, last
    #offset=k means positions at least have k move, k>=0, last move
    @staticmethod
    def to_positions(sgfGame, ret, offset):
        game=re.findall(SgfUtil.pattern, sgfGame)
        x=""
        for i, rawMove in enumerate(game):
            x +=rawMove[1:]+" "
            if i < offset:
                continue
            ret.append(x.strip())
        return ret

    @staticmethod
    def write_to_file(str_positions, filename):
        out=open(filename, "a")
        for posi in str_positions:
            out.write(posi)
            out.write('\n')
        out.close()

    #convert all .sgf games in a directory
    @staticmethod
    def process_convert(pathname, outfilename, offset):
        onlyfiles = [f for f in os.listdir(pathname) if os.path.isfile(os.path.join(pathname, f))]
        for f in onlyfiles:
            infile=open(os.path.join(pathname, f), "r")
            sgfGame=infile.read()
            positions=[]
            SgfUtil.to_positions(sgfGame, positions, offset)
            SgfUtil.write_to_file(positions, outfilename)
            infile.close()

    #remove duplicate and save in a new file
    @staticmethod
    def remove_duplicates_and_write(position_file_name, boardsize):
        tt={}
        hashUtil=ZobristHash(boardsize=boardsize)
        with open(position_file_name) as f:
            for line in f:
                line=line.strip()
                movesquence=line.split()
                intmoveseq=[]
                for m in movesquence:
                    move=m[2:4]
                    x=ord(move[0].lower())-ord('a')
                    y=ord(move[1])-ord('0')-1
                    intmoveseq.append(x*boardsize+y)
                code=hashUtil.get_hash(intmoveseq)
                tt[code]=line

        outfilename=position_file_name+"_no_duplicates"
        print("size: ", len(tt.values()))
        with open(outfilename, "w+") as f:
            for line in tt.values():
                print(line)
                f.write(line)
                f.write('\n')
def test():
    sgf = ""
    with open("0000.sgf") as f:
        sgf = f.read()
    print(sgf)
    position_list = []
    SgfUtil.to_positions(sgf, position_list, 1)
    print(position_list)

def process0():
    src_dir = "/home/cgao/Documents/hex_data/8x8"
    output = "8x8-raw0.txt"
    SgfUtil.process_convert(src_dir, output, offset=1)
    SgfUtil.remove_duplicates_and_write(output, boardsize=8)

def process1():
    src_dir="/home/cgao/Documents/hex_data/8x8-1-1/jobs/8x8-mohex-mohex-10-a-vs-mohex-mohex-10-b"
    output="8x8-raw1.txt"
    SgfUtil.process_convert(src_dir, output, offset=1)
    SgfUtil.remove_duplicates_and_write(output, boardsize=8)
def process2():
    src_dir = "/home/cgao/Documents/hex_data/8x8-1-1/jobs/8x8-mohex-mohex-cg2010-vs-wolve-wolve-cg2010"
    output = "8x8-raw2.txt"
    SgfUtil.process_convert(src_dir, output, offset=1)
    SgfUtil.remove_duplicates_and_write(output, boardsize=8)


def process3():
    src_dir = "/home/cgao/Documents/hex_data/8x8-1-1/2/jobs/8x8-mohex-mohex-16-vs-mohex-mohex-8"
    output = "8x8-raw4.txt"
    SgfUtil.process_convert(src_dir, output, offset=1)
    SgfUtil.remove_duplicates_and_write(output, boardsize=8)

def process_from_cmd():
    parser=argparse.ArgumentParser(description="Use SgfUtils to convert sgf files into state-action data or state-value data")


if __name__ == "__main__":
    #process3()
    SgfUtil.remove_duplicates_and_write("tmp2.txt", boardsize=8)
    #data_dir="/home/cgao/Documents/hex_data/8x8-1-1/jobs/8x8-mohex-mohex-10-a-vs-mohex-mohex-10-b"
    #outfile1="8x8.raw2"
    #SgfUtil.process_convert(data_dir, outfile1)
    #SgfUtil.remove_duplicates_and_write(outfile1, boardsize=8)
