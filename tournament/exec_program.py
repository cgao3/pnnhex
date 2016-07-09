#!/usr/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
sys.path.append("..")
from gtpinterface import GTPInterface
from neuralnet_agent import *
def main():
    """
    :return:
    """
    model="../savedPGModel/model.ckpt-9"
    agent=NetworkAgent(model, name="test agent")
    interface=GTPInterface(agent)
    while True:
        command=raw_input()
        success, response =interface.send_command(command)
        print("= " if success else "? ", response, "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()