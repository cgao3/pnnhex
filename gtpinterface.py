from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")

from game_util import *

class GTPInterface(object):

    def __init__(self, agent):
        self.agent=agent
        commands={}
        commands["name"]=self.gtp_name
        commands["genmove"]=self.gtp_genmove
        commands["quit"]=self.gtp_quit
        commands["showboard"]=self.gtp_show
        commands["play"]=self.gtp_play
        commands["list_commands"]=self.gtp_list_commands
        commands["clear_board"]=self.gtp_clear
        commands["gamestatus"]=self.gtp_gamestatus
        commands["close"]=self.gtp_close

        self.commands=commands

    def send_command(self, command):
        p=command.split()
        func_key=p[0]
        args=p[1:]
        assert(func_key in self.commands.keys())
        #call that function with parameters
        return self.commands[func_key](args)

    def gtp_name(self, args):
        return True, self.agent.agent_name

    def gtp_list_commands(self, args):
        return True, self.commands.keys()

    def gtp_quit(self, args):
        if self.agent.sess!=None:
            self.agent.sess.close()
        sys.exit()

    def gtp_clear(self, args):
        self.agent.reinitialize()
        return True,""

    def gtp_play(self, args):
        #play black/white a1
        assert(len(args)==2)
        intmove=raw_move_to_int(args[1])
        assert(0 <= intmove <BOARD_SIZE*BOARD_SIZE)
        black_player=0
        white_player=1
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
        :param args:
        :return:
        """
        raw_move=self.agent.generate_move()

        assert(args[0][0]=='b' or args[0][0]=='w')
        x=args[0:]
        x.append(raw_move)
        self.gtp_play(x)
        return True, raw_move

    def gtp_show(self, args):
        return True, state_to_str(self.agent.game_state)

    def gtp_close(self, args):
        try:
            self.agent.sess.close()
            return True, ""
        except AttributeError:
            return True, ""

    def gtp_gamestatus(self, args):
        status=self.agent.gamestatus()
        if status==-1:
            return True, "UNSETTLED"
        elif status==0:
            return True, "BLACK WIN"
        elif status==1:
            return True, "WHITE WIN"
        else: return False, "ERROR GAME STATUS"
