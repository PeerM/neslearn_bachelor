import socket

class Emu:
    def __init__(self, soc=None):
        self.soc = soc
        if not soc:
            self.soc = socket.create_connection(("localhost", 9090))
        self.soc_file = soc.makefile()

    def _send_command(self, command):
        self.soc.send((command + "\n").encode())

    def _send_command_with_args(self, *args):
        self.soc.send((" ".join(args) + "\n").encode())

    def _recv_feedback(self):
        return self.soc_file.readline()

    def step(self):
        self._send_command("step")
        return self._recv_feedback()

    def close(self):
        self._send_command("close")
        return self._recv_feedback()

    def get_ram(self):
        self._send_command("get_ram")
        return self.soc.recv(2048)

    def play_beginning(self):
        self._send_command("playbeginning")
        return self._recv_feedback()

    # mode supports normal, turbo or maximum
    def speed_mode(self, mode):
        self._send_command_with_args("speedmode", mode)
        return self._recv_feedback()

    def play_movie(self):
        self._send_command("play_movie")
        return self._recv_feedback().strip() == "true"
