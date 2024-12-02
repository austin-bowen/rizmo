import asyncio
from argparse import Namespace
from dataclasses import dataclass
from io import BytesIO
from threading import Thread

from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever
from flask import Flask, render_template_string, send_file

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.signal import graceful_shutdown_on_sigterm

app = Flask(__name__)


@dataclass
class Cache:
    image_bytes: bytes = b''


cache = Cache()


@app.route('/')
def index():
    return render_template_string('''
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <title>Image Display</title>
            <style>
              body {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
              }
              img {
                max-width: 100%;
                max-height: 100%;
              }
            </style>
            <script>
              function refreshImage() {
                const img = document.getElementById('image');
                img.src = '/image?' + new Date().getTime();
              }
              setInterval(refreshImage, 1000);
            </script>
          </head>
          <body>
            <img id="image" src="/image" alt="Image">
          </body>
        </html>
    ''')


@app.route('/image')
def image():
    return send_file(BytesIO(cache.image_bytes), mimetype='image/jpeg')


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    async def handle_image(topic, data):
        timestamp, camera_index, image_bytes = data
        cache.image_bytes = image_bytes

    await node.listen('new_image', handle_image)

    Thread(target=app.run, kwargs=dict(host='0.0.0.0')).start()

    await forever()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser('website')
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
