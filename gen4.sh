#!/usr/bin/env bash
# export KERAS_BACKEND=tensorflow
export PYTHONPATH=./:./extern/kerlym/:./extern/nes-async-rl:$PYTHONPATH
python3 extern/nes-async-rl/a3c_nes.py /home/peer/mario.nes --processes 2