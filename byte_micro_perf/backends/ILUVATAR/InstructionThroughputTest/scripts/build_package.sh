#!/bin/bash
SCRIPTPATH=$(dirname $(realpath "$0"))
cd $(dirname $(realpath "$0"))/..
MAX_JOBS=${MAX_JOBS:-$(nproc --all)}
PYTHON_PATH=$(which python3)
TORCH_DIR=$(echo `python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))"`)
export TORCH_DIR=$TORCH_DIR
export MAX_JOBS=${MAX_JOBS}

echo "Python cmd1: ${PYTHON_PATH} setup.py build"
${PYTHON_PATH} setup.py build 2>&1 | tee compile.log; [[ ${PIPESTATUS[0]} == 0 ]] || exit

echo "Python cmd2: ${PYTHON_PATH} setup.py bdist_wheel -d build_pip"
${PYTHON_PATH} setup.py bdist_wheel -d build_pip || exit

# Return 0 status if all finished
exit 0
