from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import  re

#Smart Game Format
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


if __name__ == "__main__":
    x="" \
      "ile-name jobs/8x8-mohex-mohex-cg2010-vs-wolve-wolve-cg2010/mohex-mohex-" \
      "cg2010-0.log Result according to W: W+] " \
      ";B[a1] " \
      ";W[d5] "

    ret=[]
    SgfUtil.to_positions(x,ret)
    print(ret)