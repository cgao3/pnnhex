from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
sys.path.append("..")
from game_util import *
import argparse
from program import Program
import threading
import time
import sys


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
    winner=None

    black_agent.sendCommand("clear_board")
    white_agent.sendCommand("clear_board")

    black=0
    white=1
    while True:
        move =black_agent.sendCommand("genmove black").strip()
        if move == "resign":
            winner=white
            return winner
        game.append(raw_move_to_int(move))
        white_agent.sendCommand("play black "+move)
        if verbose:
            pass

        sys.stdou.flush()
        move=white_agent.sendCommand("genmove white").strip()
        if move=="resign":
            winner=black
            return winner
        game.append(raw_move_to_int(move))
        black_agent.sendCommand("play white "+move)
        if verbose:
            pass
        sys.stdout.flush()

    winner_name = "black " if winner==black else "white"
    print("WIN BY "+winner_name)
    return winner

if __name__ == "__main__":
    #parser=argparse.ArgumentParser(description="tournament between agents")
    #parser.add_argument("num_games", type=int, help="num of paired games playing")
    #parser.add_argument("--verbose","-v",action="stroe_consnat", const=True, default=False, help="verbose or not")

    #args=parser.parse_args()
    num_games=1
    think_time=10
    net_exe="./exec_program.py 2>/dev/null"
    wolve_exe="/Users/gc/benzene/src/wolve/wolve 2>/dev/null"
    wolve=wrapper_agent(wolve_exe)
    net=wrapper_agent(net_exe)
    wolve.sendCommand("param_wolve max_time "+think_time)

    for i in range(num_games):
        wolve.reconnect()
        wolve.sendCommand("param_wolve max_time "+think_time)
        wolve.sendCommand("boardsize "+BOARD_SIZE)
        win=run_single_match(wolve, net)
    net.sendCommand("quit")


