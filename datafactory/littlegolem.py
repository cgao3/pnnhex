import re
import os
class LittleGolem:
    BLACK="B"
    WHITE="W"
    EMPTY=None
    def __init__(self, srcDir, srcOutbasename):
        self.srcdir=srcDir
        self.outfilename=srcOutbasename
        pass

    @staticmethod
    def convertRawGamesToRMLPositions(srcRawLgGameFileName, boardsize):
        print("converting to position-action and position-value")
        paOut=open("/tmp/pa"+repr(boardsize)+".txt","w")
        pvOut=open("/tmp/pv"+repr(boardsize)+".txt", "w")
        with open(srcRawLgGameFileName, "r") as fin:
            for game in fin:
                x = ""
                y= ""
                moveseq=game.strip().split()
                reward=float(moveseq[-1])
                for i, rawMove in enumerate(moveseq[:-1]):
                    x += rawMove + " "
                    paOut.write(x.strip()+'\n')
                    y += rawMove + " "
                    if i%2==len(moveseq[:-1])%2:
                        pvOut.write((y+repr(reward)).strip())
                        pvOut.write('\n')
                    else:
                        v=-reward
                        pvOut.write((y+repr(v)).strip())
                        pvOut.write('\n')
        paOut.close()
        pvOut.close()

    def processAllInputFilesInDir(self):
        self.fout11x11=open(self.outfilename+"raw11x11.txt","w")
        self.fout13x13 = open(self.outfilename+"raw13x13.txt", "w")
        self.fout19x19 = open(self.outfilename+"raw19x19.txt", "w")
        onlyfiles = [filename for filename in os.listdir(self.srcdir) if os.path.isfile(os.path.join(self.srcdir, filename))]
        for fname in onlyfiles:
            self.readLittleGolemAndProcess(os.path.join(self.srcdir, fname))

        self.fout11x11.close()
        self.fout13x13.close()
        self.fout19x19.close()

    def readLittleGolemAndProcess(self, filename):
        fin=open(filename,"r")
        cnt=0
        for line in fin:
            line=line.strip()
            if(line==''): continue
            print(line)
            cnt+=1
            gameresult=self.getGameResult(line)
            if not gameresult or gameresult==LittleGolem.EMPTY:
                continue
            self.checkSwap(line)
            self.getBoarSize(line)
            if(self.boardsize==11 or self.boardsize==13 or self.boardsize==19):
                moveseq, value=self.getAlternatingGame(line,gameresult,self.hasswap)
                if not moveseq:
                    continue
                if self.boardsize==11:
                    self.fout11x11.write(" ".join(moveseq) + " " + str(value) + '\n')
                elif self.boardsize==13:
                    print("move seq type:", type(moveseq))
                    self.fout13x13.write(" ".join(moveseq) + " " + str(value)+ '\n')
                elif self.boardsize==19:
                    self.fout19x19.write(" ".join(moveseq) + " " + str(value)+ '\n')
                pass
            else:
                print("no board size?? wrong!")
                break
            if cnt>100:
                break
        fin.close()
        pass

    def getAlternatingGame(self, line, RE, withSwap=False):
        assert(self.boardsize)
        assert(RE)
        movepattern=r';[B|W]\[[a-z][a-z]\]'
        moveP13=r';[B|W]\[[a-m][a-m]\]'
        moveP19=r';[B|W]\[[a-s][a-s]\]'
        moveP11=r';[B|W]\[[a-k][a-k]\]'
        if self.boardsize==11:
            movepattern=moveP11
        elif self.boardsize==13:
            movepattern=moveP13
        elif self.boardsize==19:
            movepattern=moveP19

        g=re.findall(movepattern,line)
        if len(g)<5:
            return None,None
        moveseq=[]
        if not withSwap:
            for m in g:
                # in ;B[ie] or ;W[ie]
                player=m[1]
                assert(player=='B' or player=='W')
                move=m[3:-1]
                x,y=self.convertMove(move)

                #let B be first move
                player='W' if player=='B' else 'B'
                strmove=player+'['+x+y+']'
                moveseq.append(strmove)
        else:
            assert(g[0][1]=='W' and g[1][1]=='W')
            for i,m in enumerate(g):
                if i==0:
                    x,y=self.convertMove(m[3:-1])
                    moveseq.append('B['+x+y+']')
                    continue
                player = m[1]
                assert (player == 'B' or player == 'W')
                move = m[3:-1]
                x, y = self.convertMove(move)
                # because of swap, B is the first player
                strmove = player + '[' + x + y + ']'
                moveseq.append(strmove)
        valueLabel = 1.0

        print("move seq:", moveseq, valueLabel)
        return moveseq, valueLabel

    def convertMove(self, move):
        #convert move in 'ia' style to i1
        x=move[0]
        y=ord(move[1])-ord('a')+1
        return (x,str(y))

    def getGameResult(self, line):
        patternRE=r'RE\[[B|W]\]'
        patternResign=r';[B|W]\[resign\]'
        #x=re.findall(patternRE,line)
        res=re.findall(patternResign, line)
        if not res:
            return LittleGolem.EMPTY
        elif str(res).find('W'):
            return LittleGolem.WHITE
        elif str(res).find('B'):
            return LittleGolem.BLACK
        print("RE ERROR!")
        return LittleGolem.EMPTY

    def checkSwap(self, lineGame):
        patternSwap=r';[B|W]\[swap\]'
        x=re.findall(patternSwap,lineGame)
        if x:
            self.hasswap=True
        else:
            self.hasswap=False
        print(x)

    def getBoarSize(self, lineGame):
        patterSZ13=r'SZ\[13\]'
        patterSZ11=r'SZ\[11\]'
        patterSZ19=r'SZ\[19\]'

        match11=re.findall(patterSZ11, lineGame)
        match13=re.findall(patterSZ13, lineGame)
        match19=re.findall(patterSZ19, lineGame)

        print("match 11:", match11, "\t match 13: ", match13, "\t match 19: ", match19)
        if match11:
            self.boardsize=11
        if match13:
            assert(not match11)
            self.boardsize=13
        if match19:
            assert(not match13)
            self.boardsize=19

    def firstPlayer(self):
        pass


if __name__ =="__main__":
    lg=LittleGolem(srcDir="lgData/11and13", srcOutbasename="lgData/refined/out")
    lg.processAllInputFilesInDir()
    #LittleGolem.convertRawGamesToRMLPositions("lgData/refined/outraw13x13.txt", 13)
    #LittleGolem.convertRawGamesToRMLPositions("lgData/refined/outraw11x11.txt", 11)