from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from zobrist.zobrist import HexColor
BOARD_SIZE=4

NORTH_EDGE=-1
SOUTH_EDGE=-2
WEST_EDGE=-3
EAST_EDGE=-4


class Node:
    def __init__(self, code,phi, delta, parents=None):
        self.delta=delta
        self.phi=phi
        self.code=code
        self.parents=parents

class FNode:
    def __init__(self, phi, delta, isexpanded, parents=None, children=None, estimated_updegree=None):
        self.phi = phi
        self.delta = delta
        self.isexpanded = isexpanded
        self.parents = parents
        self.children = children
        self.estimated_updegree=estimated_updegree

    def asExpanded(self):
        self.isexpanded = True


#0 for black win, 1-white win, -1 unsettled.
def winner(black_group, white_group):
    if (black_group.connected(NORTH_EDGE, SOUTH_EDGE)):
        return HexColor.BLACK
    elif (white_group.connected(WEST_EDGE, EAST_EDGE)):
        return HexColor.WHITE
    else:
        return HexColor.EMPTY

def updateUF(board, black_group, white_group, intmove, player):
    assert(player == HexColor.BLACK or player== HexColor.WHITE)
    x, y = intmove // BOARD_SIZE, intmove % BOARD_SIZE
    neighbors = []
    pattern = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, 1), (1, -1)]
    for p in pattern:
        x1, y1 = p[0] + x, p[1] + y
        if 0 <= x1 < BOARD_SIZE and 0 <= y1 < BOARD_SIZE:
            neighbors.append((x1, y1))
    if (player == HexColor.BLACK):
        if (y == 0):
            black_group.join(intmove, NORTH_EDGE)
        if (y == BOARD_SIZE - 1):
            black_group.join(intmove, SOUTH_EDGE)

        for m in neighbors:
            m2 = m[0] * BOARD_SIZE + m[1]
            if (m2 in board and list(board).index(m2) % 2 == player-1):
                black_group.join(m2, intmove)
    else:

        if (x == 0):
            white_group.join(intmove, WEST_EDGE)
        if (x == BOARD_SIZE - 1):
            white_group.join(intmove, EAST_EDGE)

        for m in neighbors:
            im = m[0] * BOARD_SIZE + m[1]
            if (im in board and list(board).index(im) % 2 == player-1):
                white_group.join(im, intmove)
    # print(black_group.parent)
    return (black_group, white_group)