import signal
from subprocess import Popen, TimeoutExpired


class ProcessManager:
    def __init__(self, python_exe: str = 'python'):
        self.python_exe = python_exe
        self.processes: list[Popen] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def popen(self, *args, **kwargs) -> Popen:
        return self.add(Popen(*args, **kwargs))

    def add(self, process: Popen) -> Popen:
        self.processes.append(process)
        return process

    def start_python_module(self, module: str, *args, cpu: int = None) -> Popen:
        return self.start_python('-m', module, *args, cpu=cpu)

    def start_python(self, *args, cpu: int = None) -> Popen:
        return self.start_on_cpu(self.python_exe, *args, cpu=cpu)

    def start_on_cpu(self, cmd, *args, cpu: int = None) -> Popen:
        cpu = ['taskset', '-c', str(cpu)] if cpu is not None else []
        return self.popen(cpu + [cmd, *args])

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

        if procs_to_kill:
            for process in procs_to_kill:
                print(f'Killing process {process.pid}')
                process.kill()

            for process in procs_to_kill:
                process.wait()
