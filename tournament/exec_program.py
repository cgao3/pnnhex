from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
sys.path.append("..")
from gtpinterface import gtpinterface
from neuralnet_agent import *
def main():
    """

    :return:
    """
    agent=network_agent("None", name="test agent")
    interface=gtpinterface(agent)
    while True:
        command=raw_input()
        success, response_move =interface.send_command(command)
        if not success: print("invalid command")
        print(response_move, "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    print("hello")
    main()