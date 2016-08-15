import re
import socket


def _input_py_to_lua(input_dict):
    return "{" + ", ".join([key + "=" + str(value).lower() for (key, value) in input_dict.items()]) + "}"


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

    def input_read(self):
        self._send_command("send_feedback(joypad.read(1))")
        received = self._recv_feedback()
        return {pair[0]: pair[1] == "true" for pair in re.findall(r"(\w*)=(true|false)", received)}

    def input_write(self, input_dict):
        lua_side = _input_py_to_lua(input_dict)
        self._send_command("joypad.write(1,{}); pos_feedback()".format(lua_side))
        return self._recv_feedback()

    def load_slot(self, slot):
        self._send_command("load_slot({}); pos_feedback()".format(slot))
        return self._recv_feedback()

    def load_slot_async(self, slot):
        self._send_command("load_slot({})".format(slot))
        return True

    def play_movie_with_output(self):
        # sanity hack, to make should nothing is left in the buffer
        # prim_soc.recv(2**10)
        self.play_beginning()
        self.speed_mode("turbo")
        ram_frame_list = list()
        inputs_list = list()
        last_state = True
        while last_state:
            last_state = self.play_movie()
            ram_frame_list.append(self.get_ram())
            inputs_list.append(self.input_read())
        return ram_frame_list, inputs_list

    def full_step(self, py_input):
        self._send_command("full_step({})".format(_input_py_to_lua(py_input)))
        return self._receive_extra_check()

    def unpause(self):
        self._send_command("emu.unpause()")

    def pause(self):
        self._send_command("emu.pause()")

    def softreset(self):
        self._send_command("emu.softreset()")

    def poweron(self):
        self._send_command("emu.poweron()")

    """This is kind of a hack because recv has returned less than 2048 in the past"""
    def _receive_extra_check(self):
        accumulator = bytes()
        while len(accumulator) < 2048:
            accumulator += self.soc.recv(2048 - len(accumulator))
        return accumulator
