import socket


class Emu2:
    def __init__(self, soc=None):
        self.soc = soc
        if not soc:
            self.soc = socket.create_connection(("localhost", 9090))
        self.soc_file = soc.makefile()

    def _send_command(self, command):
        self.soc.send((command + "\n").encode())

    # questionable if still needed
    def _send_command_with_args(self, command, *args):
        self.soc.send((command + "(" + ", ".join(args) + ")" + "\n").encode())

    def _recv_feedback(self):
        return self.soc_file.readline()

    def step(self):
        self._send_command("step()")
        return self._recv_feedback()

    def close(self):
        self._send_command("close")
        return self._recv_feedback()

    def get_ram(self):
        self._send_command("get_ram()")
        return self.soc.recv(2048)

    def play_beginning(self):
        self._send_command("movie.playbeginning();pos_feedback()")
        return self._recv_feedback()

    # mode supports normal, turbo or maximum
    def speed_mode(self, mode):
        self._send_command('emu.speedmode("{}");pos_feedback()'.format(mode))
        return self._recv_feedback()

    def play_movie(self):
        self._send_command("play_movie_frame()")
        return self._recv_feedback().strip() == "true"

    def play_movie_serverside(self):
        self._send_command("play_movie_complete()")
        buffer = b''
        self.soc.settimeout(2)
        try:
            while 1:
                received = self.soc.recv(2048)
                buffer += received
        except socket.timeout:
            pass
        self.soc.settimeout(None)
        return buffer

    def message(self, message):
        self._send_command('emu.message("{}")'.format(message))
