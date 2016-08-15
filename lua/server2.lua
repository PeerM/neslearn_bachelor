-- load namespace
local socket = require("socket");
-- create a TCP socket and bind it to the local host, at any port
local server = assert(socket.bind("*", 9090))
-- find out which port the OS chose for us
local ip, port = server:getsockname()
-- print a message informing what's up
emu.print("Please telnet to localhost on port " .. port)
emu.frameadvance()
-- wait for a connection from any client
local client = server:accept()

-- block with functions that need the client in the closure
function pos_feedback()
    client:send("true\n")
end

function send_feedback(value)
    client:send(tostring(value) .. "\n")
end

function step()
    emu.frameadvance()
    pos_feedback()
end

function is_movie_in_progress()
    return movie.length() - movie.framecount() > 0
end

function play_movie_frame()
    emu.frameadvance()
    send_feedback(is_movie_in_progress())
end

function get_ram()
    client:send(memory.readbyterange(0, 2048))
end

function _play_movie()
    emu.frameadvance()
    return is_movie_in_progress()
end

function play_movie_complete()
    while _play_movie() do
        get_ram()
    end
end

function load_slot(slot)
    local save = savestate.object(slot)
    savestate.load(save)
end

function full_step(inputs)
    joypad.write(1, inputs)
    emu.frameadvance()
    get_ram()
end

function step_repeat_actions(inputs,repeat_actions)
    for variable = 0, repeat_actions, 1 do
        joypad.write(1, inputs)
        emu.frameadvance()
    end
    get_ram()
end

-- make sure we don't block waiting for this client's line
client:settimeout(10000)
-- loop forever waiting for clients
while 1 do
    -- receive the line
    local line, err = client:receive()
    -- if there was no error, send it back to the client
    if not err then
        if line == "close" then
            pos_feedback()
            break
        else
            local f = loadstring(line)
            f()
        end
    else
        emu.print(err)
        break
    end
end
client:close()