from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from program import Program
import sys
sys.path.append("..")
from game_util import *
import argparse
import threading
import time
from unionfind import unionfind


class wrapper_agent(object):

    def __init__(self, executable):
        self.executable=executable
        self.program=Program(self.executable, True)
        self.name=self.program.sendCommand("name").strip()
        self.lock=threading.Lock()

    def sendCommand(self, command):
        self.lock.acquire()
        answer=self.program.sendCommand(command)
        self.lock.release()
        return answer

    def reconnect(self):
        self.program.terminate()
        self.program=Program(self.executable, True)
        self.lock=threading.Lock()



def run_single_match(black_agent, white_agent, verbose=False):
    game=[]
    black_agent.sendCommand("clear_board")
    white_agent.sendCommand("clear_board")
    black_groups=unionfind()
    white_groups=unionfind()
    turn=0
    gamestatus=-1
    while gamestatus==-1:
        if turn==0:
            move = black_agent.sendCommand("genmove black").strip()
            if move == "resign":
                return 1
            white_agent.sendCommand("play black "+move)
        else:
            move=white_agent.sendCommand("genmove white").strip()
            if move=="resign":
                return 0
            black_agent.sendCommand("play white "+move)
        imove=raw_move_to_int(move)
        game.append(imove)

        black_groups, white_groups=update_unionfind(imove, turn, game, black_groups, white_groups)
        gamestatus=winner(black_groups,white_groups)
        if verbose:
            print(state_to_str(game))
        turn=(turn+1)%2
        sys.stdout.flush()
    return gamestatus

if __name__ == "__main__":
    #parser=argparse.ArgumentParser(description="tournament between agents")
    #parser.add_argument("num_games", type=int, help="num of paired games playing")
    #parser.add_argument("--verbose","-v",action="stroe_consnat", const=True, default=False, help="verbose or not")

    #args=parser.parse_args()
    num_games=10
    think_time=5
    net_exe="./exec_program.py 2>/dev/null"
    wolve_exe="/home/cgao3/benzene/src/wolve/wolve 2>/dev/null"
    wolve=wrapper_agent(wolve_exe)
    net=wrapper_agent(net_exe)
    wolve.sendCommand("param_wolve max_time "+repr(think_time))
    white_win_count=0
    black_win_count=0
    for i in range(num_games):
        wolve.reconnect()
        wolve.sendCommand("param_wolve max_time "+ repr(think_time))
        wolve.sendCommand("boardsize "+ repr(BOARD_SIZE))
        win=run_single_match(net, wolve, True)
        if win==0: black_win_count += 1
        else: white_win_count +=1
    net.sendCommand("close")
    print("black win ", black_win_count, "white win count ", white_win_count)
