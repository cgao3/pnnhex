
class gtpinterface(object):


    def __init__(self, agent):
        self.agent=agent
        commands={}
        commands["name"]=self.gtp_name
        commands["genmove"]=self.gtp_genmove
        commands["quit"]=self.gtp_quit
        commands["boardsize"]=self.gtp_boardsize
        commands["showboard"]=self.gtp_show

        self.commands=commands
        pass

    def send_command(self, command):
        pass

    def register_command(self, name, command):
        pass

    def gtp_name(self, args):
        pass

    def gtp_version(self, args):
        pass

    def gtp_protocol(self, args):
        pass

    def gtp_known(self, args):
        pass

    def gtp_quit(self, args):
        pass

    def gtp_boardsize(self, args):
        pass

    def gtp_clear(self, args):
        pass

    def gtp_undo(self, args):
        pass

    def gtp_play(self, args):
        pass

    def gtp_genmove(self, args):
        pass

    def gtp_time(self, args):
        pass

    def gtp_show(self, args):
        pass

    def gtp_winner(self, args):
        pass

