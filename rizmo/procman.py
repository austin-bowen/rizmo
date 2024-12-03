import signal
from dataclasses import dataclass
from subprocess import Popen, TimeoutExpired


class ProcessManager:
    def __init__(self, python_exe: str = 'python'):
        self.python_exe = python_exe

        self._options = Options()
        self.processes: list[Popen] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def options(self, **kwargs) -> 'OptionsContext':
        return OptionsContext(self, Options(**kwargs))

    def popen(self, *args, **kwargs) -> Popen:
        cpu = self._options.cpu
        cpu = ['taskset', '-c', str(cpu)] if cpu is not None else []

        args = cpu + list(args)

        shell = self._options.shell
        if shell:
            args = [' '.join(args)]

        return self.add(Popen(args, shell=shell, **kwargs))

    def add(self, process: Popen) -> Popen:
        self.processes.append(process)
        return process

    def start_python(self, *args, **kwargs) -> Popen:
        return self.popen(self.python_exe, *args, **kwargs)

    def stop(self, timeout=10.):
        print(f'Stopping {len(self.processes)} processes...')
        for process in self.processes:
            process.send_signal(signal.SIGTERM)

        procs_to_kill = []
        for process in self.processes:
            try:
                process.wait(timeout)
            except TimeoutExpired:
                print(f'Process {process.pid} did not stop in time.')
                procs_to_kill.append(process)

        for process in procs_to_kill:
            print(f'Killing process {process.pid}')
            process.kill()

        for process in procs_to_kill:
            process.wait()

    def wait(self, timeout: float = None):
        for process in self.processes:
            process.wait(timeout)


@dataclass
class Options:
    cpu: int = None
    shell: bool = False


@dataclass
class OptionsContext:
    procman: ProcessManager
    options: Options
    prev_options: Options = None

    def __enter__(self):
        self.prev_options = self.procman._options
        self.procman._options = self.options
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.procman._options = self.prev_options
