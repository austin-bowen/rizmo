import asyncio
from argparse import Namespace

from easymesh import build_mesh_node_from_args

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import SetServoPosition
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm


def get_servo_command() -> SetServoPosition:
    commands = input('<channel> <target_deg>[,] | off: ')
    if not commands:
        return SetServoPosition()

    if commands == 'off':
        return SetServoPosition('off', 'off', 'off')

    servo_command = SetServoPosition()

    for command in commands.split(','):
        channel, target_deg = command.strip().lower().split()

        channel = int(channel)

        if target_deg != 'off':
            target_deg = float(target_deg)

        if channel == 0:
            servo_command.pan_deg = target_deg
        elif channel == 1:
            servo_command.tilt0_deg = target_deg
        elif channel == 2:
            servo_command.tilt1_deg = target_deg

    return servo_command


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    while True:
        servo_command = await asyncio.to_thread(get_servo_command)
        await node.send(Topic.SERVO_COMMAND, servo_command)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
