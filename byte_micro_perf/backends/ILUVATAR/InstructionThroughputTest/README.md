build && install:
cd bytemlperf/byte_micro_perf/backends/ILUVATAR/InstructionThroughputTest/scripts
bash build_package.sh
bash install_package.sh  
run test:
cd bytemlperf/byte_micro_perf/backends/ILUVATAR/InstructionThroughputTest/test && python3 -m unittest test.py 