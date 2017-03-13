from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")

from utils.game_util import *
from utils.read_data import BOARD_SIZE

#GtpInterface for Neural Network Agent
class GTPInterface(object):

    def __init__(self, agent):
        self.agent=agent
        commands={"name": self.gtp_name,
                  "genmove": self.gtp_genmove,
                  "quit": self.gtp_quit,
                  "showboard": self.gtp_show,
                  "play": self.gtp_play,
                  "list_commands": self.gtp_list_commands,
                  "clear_board": self.gtp_clear,
                  "boardsize": self.gtp_boardsize,
                  "close": self.gtp_close}

        self.commands=commands

    def send_command(self, command):
        p=command.split()
        func_key=p[0]
        args=p[1:]

        #ignore unknow commands
        if func_key not in self.commands:
            return True,""

        #call that function with parameters
        return self.commands[func_key](args)

    def gtp_name(self, args=None):
        return True, self.agent.agent_name

    def gtp_list_commands(self, args=None):
        return True, self.commands.keys()

    def gtp_quit(self, args=None):
        if self.agent.sess!=None:
            self.agent.sess.close()
        sys.exit()

    def gtp_clear(self, args=None):
        self.agent.reinitialize()
        return True,""

    def gtp_play(self, args):
        #play black/white a1
        assert(len(args)==2)
        intmove=MoveConvertUtil.rawMoveToIntMove(args[1])
        assert(0 <= intmove <BOARD_SIZE*BOARD_SIZE)
        black_player=HexColor.BLACK
        white_player=HexColor.WHITE
        if intmove in self.agent.game_state:
            return False, "INVALID! Occupied position."

        if args[0][0]=='b':
            self.agent.play_move(black_player, intmove)
        elif args[0][0]=='w':
            self.agent.play_move(white_player, intmove)
        else:
            return False, "player should be black/white"

        return True, ""

    def gtp_genmove(self, args):
        """
        automatically detect who is to play
        """
        assert (args[0][0] == 'b' or args[0][0] == 'w')
        black=HexColor.BLACK
        white=HexColor.WHITE
        if args[0][0]=='b':
            raw_move=self.agent.generate_move(intplayer=black)
        else: raw_move=self.agent.generate_move(intplayer=white)
        x=args[0:]
        x.append(raw_move)
        self.gtp_play(x)
        return True, raw_move

    def gtp_boardsize(self, args=None):
        return True,""

    def gtp_show(self, args=None):
        return True, state_to_str(self.agent.game_state)

    def gtp_close(self, args=None):
        try:
            self.agent.sess.close()
        except AttributeError:
            pass
        return True, ""