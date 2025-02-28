#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse


def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        # TODO: this is required.
        '-f', '--recorder_filename',
        metavar='F',
        default="recording0306.log",
        help='recorder filename (recording0306.log)')
    argparser.add_argument(
        # TODO: If you want to run a whole period, use '-a'.
        '-a', '--show_all',
        action='store_true',
        help='show detailed info about all frames content')
    argparser.add_argument(
        # TODO: this is required too.
        '-s', '--save_to_file',
        metavar='S',
        help='save result to file (specify name and extension such as \'../mycarla/logs/record0306.txt\', and path before it if you need it)')

    args = argparser.parse_args()

    try:

        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        if args.save_to_file:
            doc = open(args.save_to_file, "w+")
            doc.write(client.show_recorder_file_info(args.recorder_filename, args.show_all))
            doc.close()
        else:
            print(client.show_recorder_file_info(args.recorder_filename, args.show_all))


    finally:
        pass


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
