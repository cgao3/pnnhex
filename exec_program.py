from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
from gtpinterface import gtpinterface
def main():
    """

    :return:
    """
    interface=gtpinterface(None)
    while True:
        command=raw_input()
        success, response_move =interface.send_command(command)
        if not success: print("invalid command")
        print(response_move, "\n")
        sys.stdout.flush()


if __name__ == "main":
    main()