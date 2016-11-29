from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import  re
import os
from zobrist.zobrist import *
#Smart Game Format
''''
First need to converte a .sgf game into some positions.
A directory has many .sgf games, so do them one by one.
Save the positions in a database (a text file),
and then remove the duplicates, save to a new file.
'''
class SgfUtil:

    pattern=r';[B|W]\[[a-zA-Z][1-9]\]'

    #convert a sgf Hex game to a list of positions, last
    @staticmethod
    def to_positions(sgfGame, ret):
        game=re.findall(SgfUtil.pattern, sgfGame)
        x=""
        for i, rawMove in enumerate(game):
            toplay=rawMove[1].upper()
            move=str(rawMove[3:5]).lower()
            x = x + toplay + '[' + move + '] '
            ret.append(x)
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
    def process_convert(pathname, outfilename):
        onlyfiles = [f for f in os.listdir(pathname) if os.path.isfile(os.path.join(pathname, f))]
        for f in onlyfiles:
            infile=open(os.path.join(pathname, f), "r")
            sgfGame=infile.read()
            positions=[]
            SgfUtil.to_positions(sgfGame, positions)
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

if __name__ == "__main__":

    data_dir="/home/cgao/Documents/hex_data/8x8/"
    outfile1="8x8.raw"
    SgfUtil.process_convert(data_dir, outfile1)
    SgfUtil.remove_duplicates_and_write(outfile1, boardsize=8)
    x="" \
      "ile-name jobs/8x8-mohex-mohex-cg2010-vs-wolve-wolve-cg2010/mohex-mohex-" \
      "cg2010-0.log Result according to W: W+] " \
      ";B[a1] " \
      ";W[d5] "

    ret=[]
    SgfUtil.to_positions(x,ret)
    print(ret)
    SgfUtil.write_to_file(ret, "hello.txt")