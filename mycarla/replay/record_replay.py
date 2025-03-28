import argparse

import carla
import time


def record_world(client, filename, duration=60):
    # Start recording for 60 seconds (1 minute)
    print("Starting recording...")
    client.start_recorder(filename, True)
    time.sleep(duration)  # Record for 1 minute
    client.stop_recorder()
    print(f"Recording stopped and saved as '{filename}'")


def replay_record(client):
    # Load the recorded file
    print("Loading recording...")
    client.replay_file("recording.log", 0, 0, 0)

    # Wait for the replay to finish
    while not client.is_replaying():
        time.sleep(0.1)

    print("Replay started.")
    try:
        while client.is_replaying():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping replay.")
        client.stop_replay()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-s',
        '--server-ip',
        metavar='S',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p',
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-m',
        '--mode',
        metavar='M',
        choices=['record', 'replay'],
        default='record',
        help='Mode to run: record or replay (default: record)')
    argparser.add_argument(
        '-d',
        '--duration',
        metavar='D',
        default=60,
        type=int,
        help='Duration of recording (default: 60)')
    argparser.add_argument(
        # TODO: the highest frequency to be modified
        '-f',
        '--filename',
        metavar='F',
        default='recording0306.log',
        help='Path to recording (default: recording0306.log)')

    args = argparser.parse_args()

    # Connect to Carla server
    client = carla.Client(args.server_ip, args.port)
    client.set_timeout(10.0)

    if args.mode == 'record':
        record_world(client, args.filename, duration=args.duration)
    elif args.mode == 'replay':
        replay_record(client)


if __name__ == '__main__':
    main()
