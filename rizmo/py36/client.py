"""Client for the Python 3.6 server running on the Jetson Nano."""

import asyncio
import code
import pickle
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

from easymesh.asyncio import Reader, Writer
from typing_extensions import Callable

from rizmo.nodes.messages_py36 import Detection
from rizmo.py36.obj_detector import Image
from rizmo.py36.server import DEFAULT_SOCKET_PATH, MAX_MESSAGE_LEN, PICKLE_PROTOCOL


class Py36Client:
    def __init__(
            self,
            conn_builder: Callable[[], Awaitable['Connection']],
    ) -> None:
        self.conn_builder = conn_builder

        self._conn = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    @classmethod
    def build(
            cls,
            socket_path: str = DEFAULT_SOCKET_PATH,
    ) -> 'Py36Client':
        async def conn_builder():
            reader, writer = await asyncio.open_unix_connection(socket_path)
            return Connection(reader, writer)

        return cls(conn_builder)

    async def _get_conn(self) -> 'Connection':
        if self._conn is None:
            self._conn = await self.conn_builder()

        return self._conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def detect(self, image: Image) -> list[Detection]:
        return await self.rpc('detect', image)

    async def ping(self) -> str:
        return await self.rpc('ping')

    async def stop_server(self) -> None:
        return await self.rpc('stop_server')

    async def rpc(self, function, *args, **kwargs) -> Any:
        try:
            await self._send_rpc_request((function, args, kwargs))
            return self._read_rpc_response()
        except ConnectionError:
            await self.close()
            raise

    async def _send_rpc_request(self, rpc_request):
        conn = await self._get_conn()
        request_data = pickle.dumps(rpc_request, protocol=PICKLE_PROTOCOL)
        await write_message(conn.writer, request_data)

    async def _read_rpc_response(self) -> Any:
        conn = await self._get_conn()
        response_data = await read_message(conn.reader)
        return pickle.loads(response_data)


async def read_message(reader: Reader) -> bytes:
    message_len = int.from_bytes(
        await reader.readexactly(4),
        byteorder='big',
        signed=False,
    )

    return await reader.readexactly(message_len)


async def write_message(writer: Writer, message: bytes) -> None:
    if len(message) > MAX_MESSAGE_LEN:
        raise ValueError(
            f'Message is too large to send. Size is {len(message)} bytes; '
            f'max size is {MAX_MESSAGE_LEN} bytes.'
        )

    message_len = len(message).to_bytes(4, byteorder='big', signed=False)
    writer.write(message_len)
    writer.write(message)
    await writer.drain()


@dataclass
class Connection:
    reader: Reader
    writer: Writer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    async def close(self) -> None:
        self.writer.close()
        await self.writer.wait_closed()


async def main() -> None:
    client = Py36Client.build()

    print('Use variable `client` to interact with the server.')
    code.interact(local=locals())


if __name__ == '__main__':
    asyncio.run(main())
