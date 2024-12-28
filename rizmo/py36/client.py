"""Client for the Python 3.6 server running on the Jetson Nano."""

import code
import pickle
import socket
from typing import Any

from typing_extensions import Callable

from rizmo.nodes.messages_py36 import Detection
from rizmo.py36.obj_detector import Image
from rizmo.py36.server import DEFAULT_SOCKET_PATH, PICKLE_PROTOCOL, read_message, write_message


class Client:
    def __init__(
            self,
            conn_builder: Callable[[], 'Connection'],
    ) -> None:
        self.conn_builder = conn_builder

        self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @classmethod
    def build(
            cls,
            socket_path: str = DEFAULT_SOCKET_PATH,
    ) -> 'Client':
        def conn_builder():
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(socket_path)
            return Connection(sock)

        return cls(conn_builder)

    @property
    def conn(self) -> 'Connection':
        if self._conn is None:
            self._conn = self.conn_builder()

        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def detect(self, image: Image) -> list[Detection]:
        return self.rpc('detect', image)

    def ping(self) -> str:
        return self.rpc('ping')

    def stop_server(self) -> None:
        return self.rpc('stop_server')

    def rpc(self, function, *args, **kwargs) -> Any:
        try:
            self._send_rpc_request((function, args, kwargs))
            return self._read_rpc_response()
        except BrokenPipeError:
            self.close()
            raise

    def _send_rpc_request(self, rpc_request):
        request_data = pickle.dumps(rpc_request, protocol=PICKLE_PROTOCOL)
        write_message(self.conn.wfile, request_data)

    def _read_rpc_response(self) -> Any:
        response_data = read_message(self.conn.rfile)
        return pickle.loads(response_data)


class Connection:
    def __init__(
            self,
            sock: socket.socket,
            rbufsize: int = -1,
            wbufsize: int = 0,
    ) -> None:
        self.sock = sock
        self.rfile = sock.makefile('rb', buffering=rbufsize)
        self.wfile = sock.makefile('wb', buffering=wbufsize)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        self.sock.close()


def main() -> None:
    client = Client.build()

    print('Use variable `client` to interact with the server.')
    code.interact(local=locals())


if __name__ == '__main__':
    main()
