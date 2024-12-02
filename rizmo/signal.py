import os
import signal


def graceful_shutdown_on_sigterm():
    """
    By default, asyncio does not gracefully shut down on SIGTERM,
    but it does on SIGINT (KeyboardInterrupt). This function will
    make asyncio gracefully shut down on SIGTERM by re-emitting
    the signal as SIGINT.
    """

    def _handler(signum, frame):
        os.kill(os.getpid(), signal.SIGINT)

    signal.signal(signal.SIGTERM, _handler)
