from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from read_data import BOARD_SIZE

def int_to_pair(intmove):
    x=intmove//BOARD_SIZE
    y=intmove%BOARD_SIZE
    return (x,y)

def pair_to_int(move):
    x,y=move[0],move[1]
    return x*BOARD_SIZE+y

def state_to_str(g):
    size=BOARD_SIZE
    white = 'O'
    black = '@'
    empty = '.'
    ret = '\n'
    coord_size = len(str(size))
    offset = 1
    ret+=' '*(offset+1)
    board=[None]*size
    for i in range(size):
        board[i]=[empty]*size

    for k, i in enumerate(g):
        x,y=i//size, i%size
        board[x][y]=black if k%2==0 else white

    PLAYERS = {"white": white, "black": black}
    for x in range(size):
        ret += chr(ord('A') + x) + ' ' * offset * 2
    ret += '\n'
    for y in range(size):
        ret += str(y + 1) + ' ' * (offset * 2 + coord_size - len(str(y + 1)))
        for x in range(size):
            if (board[x][y] == PLAYERS["white"]):
                ret += white
            elif (board[x][y] == PLAYERS["black"]):
                ret += black
            else:
                ret += empty
            ret += ' ' * offset * 2
        ret += white + "\n" + ' ' * offset * (y + 1)
    ret += ' ' * (offset * 2 + 1) + (black + ' ' * offset * 2) * size

    return ret

def next_player(current_player):
    return (current_player + 1) % 2