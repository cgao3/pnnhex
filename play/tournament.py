from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from program import Program
import sys
from game_util import *
import argparse
from unionfind import unionfind
from agents import WrapperAgent

EXE_NN_AGENT_NAME="./exec_nn_agent.py "
EXE_HEX_PATH="/home/cgao3/benzene/src/wolve/wolve "

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
            move = black_agent.genmove_black()
            if move == "resign":
                print("black resign")
                print(state_to_str(game))
                return 1
            white_agent.play_black(move)
        else:
            move=white_agent.genmove_white()
            if move=="resign":
                print("white resign")
                print(state_to_str(game))
                return 0
            black_agent.play_white(move)
        imove=raw_move_to_int(move)
        black_groups, white_groups=update_unionfind(imove, turn, game, black_groups, white_groups)
        gamestatus=winner(black_groups,white_groups)
        game.append(imove)
        if verbose:
            print(state_to_str(game))
        turn=(turn+1)%2
        sys.stdout.flush()
    print("gamestatus", gamestatus)
    print(state_to_str(game))
    return gamestatus

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="tournament between agents")
    #parser.add_argument("num_games", type=int, help="num of paired games playing")
    parser.add_argument("model_path", help="where the model locates", type=str)
    parser.add_argument("--wolve_path",default="/home/cgao3/benzene/src/wolve/wolve", help="where is the wolve", type=str)
    parser.add_argument("--value_net", help="whether it is valuenet model", action="store_true", default=False)
    parser.add_argument("--verbose", help="verbose?", action="store_true", default=False)

    args=parser.parse_args()
    num_games=1000
    think_time=1
    net_exe=EXE_NN_AGENT_NAME + args.model_path +" 2>/dev/null"
    EXE_HEX_PATH=args.wolve_path
    wolve_exe=EXE_HEX_PATH+" 2>/dev/null"
    wolve=WrapperAgent(wolve_exe, True)
    net=WrapperAgent(net_exe, True)

    wolve.sendCommand("param_wolve max_time "+repr(think_time))
    white_win_count=0
    black_win_count=0
    for i in range(num_games):
        wolve.reconnect()
        wolve.sendCommand("param_wolve max_time "+ repr(think_time))
        wolve.sendCommand("boardsize "+ repr(BOARD_SIZE))
        win=run_single_match(net, wolve, False)
        if win==0: black_win_count += 1
        if win==1: white_win_count +=1
        print(i+1, "black: ", black_win_count, "white: ", white_win_count)
    net.sendCommand("close")
    print("black win ", black_win_count, "white win count ", white_win_count)
