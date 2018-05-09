# Random

"""bash
./runner.sh -c r -m random -k 50 -r 5 -t
"""

# CRP

## LSA

"""bash
./runner.sh -c c -m LSA -a 0.01 -t -n 100 -s 50 -k 0.01 -S 0.005
./runner.sh -c c -m LSA -a 0.01 -t -n 100 -s 100 -k 0.01 -S 0.01
"""

## LDA

"""bash
./runner.sh -c c -m LDA -a 0.0000005 -s 50 -r 5 -n 100 -t -k 0.01 -S 0.000001
./runner.sh -c c -m LDA -a 0.0000005 -s 100 -r 5 -n 100 -t -k 0.01 -S 0.0001
"""

## doc2vec


"""bash
./runner.sh -c c -m doc2vec -a 0.05 -s 50 -r 5 -n 100 -t -k 0.01 -S 0.25
./runner.sh -c c -m doc2vec -a 0.1 -s 100 -r 1 -n 100 -t -k 0.01 -S 0.08
"""


# ddCRP

## LSA

"""bash
./runner.sh -c d -m LSA -a 10 -s 50 -r 5 -n 25 -t -k 0.01 -S 0.01 -b -10
./runner.sh -c d -m LSA -a 1 -s 100 -r 5 -n 25 -t -k 0.01 -S 0.01 -b -26
"""

## LDA

"""bash
./runner.sh -c d -m LDA -a 1 -s 50 -r 5 -n 25 -t -k 0.01 -S 0.0001 -b -10
./runner.sh -c d -m LDA -a 500 -s 100 -r 5 -n 25 -t -k 0.01 -S 0.0001 -b -26.5
"""

## doc2vec

"""bash
./runner.sh -c d -m doc2vec -a 0.05 -s 50 -r 5 -n 25 -t -k 0.01 -S 1 -b -23
./runner.sh -c d -m doc2vec -a 1 -s 100 -r 5 -n 25 -t -k 0.01 -S 1 -b -35
"""
