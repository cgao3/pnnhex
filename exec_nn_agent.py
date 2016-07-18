#!/usr/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from gtpinterface import GTPInterface
from policygradient import PG_MODEL_DIR
import os
import sys
from agents import NNAgent

def main():
    """
    Executable of the neural net player..
    use the latest model saved by policy gradient reinforcement learning
    """

    check_point=os.path.join(".", PG_MODEL_DIR, "checkpoint")
    with open(check_point, "r") as f:
        line=f.readline()
    model_name=line.split()[1][1:-1]
    model_path=os.path.join(".", PG_MODEL_DIR, model_name)
    agent=NNAgent(model_path, name=model_name)
    interface=GTPInterface(agent)
    while True:
        command=raw_input()
        success, response =interface.send_command(command)
        print("= " if success else "? ", response, "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
