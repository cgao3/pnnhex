from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

class PNS:

    def __init__(self, state, toplay):
        self.position=state
        self.toplay=toplay
        black_stones=[state[i] for i in range(len(state)) if i%2==0]
        white_stones=[state[i] for i in range(len(state)) if i%2!=0]
        
    def _init_hash(self):
        pass


    def dfpns(self):
        pass

    def MID(self):
        pass
    def selection(self):
        pass

    def backup(self):
        pass

    def expand(self):
        pass

    def evaluate(self):
        pass
