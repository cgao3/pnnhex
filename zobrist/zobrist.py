from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

from zobrist import zobristhashes


class HexColor:
    def __init__(self):
        pass
    BLACK, WHITE, EMPTY = range(1,4)

class ZobristHash:
    def __init__(self, boardsize):
        self.m_hash=None
        self._read_predefined_hashes()

        self.m_base=self.m_hashes[boardsize*30+boardsize]
        self.m_black_hashes=self.m_hashes[1024:2048]
        self.m_white_hashes=self.m_hashes[2048:3072]
        self.color_hashes=[None, None, None]
        self.color_hashes[HexColor.BLACK]=self.m_hashes[1024:2048]
        self.color_hashes[HexColor.WHITE]=self.m_hashes[2048:3072]
        self.toplay_hashes=[None, None, None, None]
        self.toplay_hashes[HexColor.BLACK]=self.m_hashes[3072]
        self.toplay_hashes[HexColor.WHITE]=self.m_hashes[3073]
        self.toplay_hashes[HexColor.EMPTY]=self.m_hashes[3074]

        self.reset()

    def _read_predefined_hashes(self):
        sz=len(zobristhashes.predefined_hashes)
        self.m_hashes=np.ndarray(dtype=np.uint64, shape=(sz,))
        for ind, hashcode_string in enumerate(zobristhashes.predefined_hashes):
            self.m_hashes[ind]=int(hashcode_string, 16)

    def reset(self):
        self.m_hash=self.m_base

    def update(self, toplay, hexpoint):
        assert (toplay==HexColor.BLACK or
                toplay==HexColor.WHITE or toplay==HexColor.EMPTY )
        self.m_hash=self.m_hash ^ self.color_hashes[toplay][hexpoint]
        return self.m_hash

    def hash(self, toplay):
        return self.m_hash ^ self.toplay_hashes[toplay]

    def get_hash(self, intstate):
        toplay=HexColor.BLACK
        code=self.m_base
        for m in intstate:
            code = code ^ self.color_hashes[toplay][m]
            toplay = HexColor.EMPTY - toplay
        return code

    def update_hash(self, code, intmove, intplayer):

        return  code ^ self.color_hashes[intplayer][intmove]

    def compute(self, black_stone_pos, white_stone_pos, toplay=None):
        self.reset()
        for pos in black_stone_pos:
            self.m_hash = self.m_hash ^ self.m_black_hashes[pos]
        for pos in white_stone_pos:
            self.m_hash = self.m_hash ^ self.m_white_hashes[pos]
        if toplay :
            assert(HexColor.BLACK<=toplay <= HexColor.EMPTY)
            self.m_hash = self.m_hash ^ self.toplay_hashes[toplay]

        return self.m_hash

if __name__ == "__main__":
    print(len(zobristhashes.predefined_hashes))
    print(zobristhashes.predefined_hashes[0])
    print(zobristhashes.predefined_hashes[1])
