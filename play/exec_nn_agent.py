#!/usr/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from play.gtpinterface import GTPInterface
import os
import sys
from play.agents import NNAgent
import argparse

def main(argv=None):
    """
    Executable of the neural net player..
    use the latest model saved by policy gradient reinforcement learning
    """

    agent=NNAgent(argv.model_path, name=argv.model_path, is_value_net=argv.value_net)
    interface=GTPInterface(agent)
    while True:
        command=raw_input()
        success, response =interface.send_command(command)
        print("= " if success else "? ", response, "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='test')
    parser.add_argument('model_path', type=str, help="the path of the model file")
    parser.add_argument('--value_net', action='store_true', help="value_net model?")
    parser.add_argument('--verbose', action='store_true', help='verbose?')
    args=parser.parse_args()
    main(args)
