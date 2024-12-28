"""
Server for running code on the Jetson Nano that requires the native Python 3.6.
"""

import os
import pickle
import signal
import time
from argparse import ArgumentParser, Namespace
from io import BytesIO
from socketserver import StreamRequestHandler, UnixStreamServer
from threading import Thread
from typing import Any

import numpy as np

from rizmo.py36.obj_detector import ObjectDetector, get_object_detector
from rizmo.signal import graceful_shutdown_on_sigterm

DEFAULT_SOCKET_PATH: str = '/tmp/rizmo.py36_server.sock'

PICKLE_PROTOCOL: int = 4
"""The highest protocol version supported by Python 3.6."""

MAX_MESSAGE_LEN: int = 2 ** 32 - 1


def main(args: Namespace) -> None:
    object_detector = get_object_detector(args.network, args.threshold)

    rpc_handler = RpcHandler(object_detector)

    def delete_socket_file():
        try:
            os.unlink(args.socket_path)
        except FileNotFoundError:
            pass

    delete_socket_file()
    try:
        print(f'Starting server at {args.socket_path!r}...')
        with UnixStreamServer(
                args.socket_path,
                RequestHandler.builder(rpc_handler),
        ) as server:
            server.serve_forever()
    finally:
        delete_socket_file()


class RpcHandler:
    def __init__(self, object_detector: ObjectDetector):
        self.object_detector = object_detector

    def __call__(self, function_name, args, kwargs) -> Any:
        function = getattr(self, function_name)
        return function(*args, **kwargs)

    def detect(self, image_bytes: BytesIO) -> Any:
        image = np.load(image_bytes)
        return self.object_detector.get_objects(image)

    def ping(self) -> str:
        return 'pong'

    def stop_server(self) -> None:
        print('Stopping server...')

        def stop():
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGINT)

        Thread(target=stop, daemon=True).start()


class RequestHandler(StreamRequestHandler):
    def __init__(self, rpc_handler: RpcHandler, *args):
        self.rpc_handler = rpc_handler
        super().__init__(*args)

    @classmethod
    def builder(cls, model):
        def builder(*args) -> RequestHandler:
            return cls(model, *args)

        return builder

    def handle(self) -> None:
        try:
            while True:
                self._handle_one_request()
        except EOFError:
            pass

    def _handle_one_request(self) -> None:
        request = self._get_rpc_request()
        response = self.rpc_handler(*request)
        self._write_response(response)

    def _get_rpc_request(self):
        request_data = read_message(self.rfile)
        return pickle.loads(request_data)

    def _write_response(self, response: Any) -> None:
        response_data = pickle.dumps(response, protocol=PICKLE_PROTOCOL)
        write_message(self.wfile, response_data)


def read_message(rfile) -> bytes:
    message_len = int.from_bytes(
        rfile.read(4),
        byteorder='big',
        signed=False,
    )

    return rfile.read(message_len)


def write_message(wfile, message: bytes) -> None:
    if len(message) > MAX_MESSAGE_LEN:
        raise ValueError(
            f'Message is too large to send. Size is {len(message)} bytes; '
            f'max size is {MAX_MESSAGE_LEN} bytes.'
        )

    message_len = len(message).to_bytes(4, byteorder='big', signed=False)
    wfile.write(message_len)
    wfile.write(message)
    wfile.flush()


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        '--socket-path',
        default=DEFAULT_SOCKET_PATH,
        help='Path to the Unix socket. Default: %(default)s',
    )

    parser.add_argument(
        '--network',
        default='ssd-mobilenet-v2',
        help='Pre-trained model to load. Default: %(default)s',
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Minimum detection threshold to use. Default: %(default)s',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    main(parse_args())
