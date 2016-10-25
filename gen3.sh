export KERAS_BACKEND=tensorflow
export PYTHONPATH=./:./extern/kerlym/:$PYTHONPATH
python3 hsa/gen3/mario_kerlym.py -P --plot_rate 30 -c 15 -R