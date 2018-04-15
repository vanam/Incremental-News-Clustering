#!/bin/bash

######################
# Autor: Martin Váňa #
######################

################################################################################
# FLAGS                                                                        #
################################################################################

TEST=0

################################################################################
# VARIABLES                                                                    #
################################################################################

K=5
CLUSTERING=""
MODEL=""
REPETITIONS=5
SIZE=100

################################################################################
# FUNCTIONS                                                                    #
################################################################################

function random {
    ./run.sh clustering_system/main.py -c news -m random -l dummy -K ${K} -s ${SIZE} -i $1
}

function bgmmm {
    ./run.sh clustering_system/main.py -c news -m ${MODEL} -l BGMM -K ${K} -s ${SIZE} -i $1
}

function crp {
    ./run.sh clustering_system/main.py -c news -m ${MODEL} -l CRP -s ${SIZE} -i $1
}

function ddcrp {
    ./run.sh clustering_system/main.py -c news -m ${MODEL} -l ddCRP -s ${SIZE} -i $1
}

function help {
    echo "Run clustering experiments"
    echo "--------------------------"
    echo "Copyright (c) Martin Váňa, 2018"
    echo ""
    echo "Usage: $1 [OPTIONS]"
    echo "Available options"
    echo "  -c <method={r|b|c|d}>                use clustering method where:"
    echo "                                           -r    random"
    echo "                                           -b    BGMM"
    echo "                                           -c    CRP"
    echo "                                           -d    ddCRP"
    echo "  -h                                   displays help"
    echo "  -k <number>                          the number of clusters (if applicable)"
    echo "  -m <model={random|LSI|LDA|doc2vec}>  use specified model"
    echo "  -n <number>                          the number of repetitions"
    echo "  -s <size>                            the size of a feature vector"
    echo "  -t                                   use test data"
}

################################################################################
# CODE                                                                         #
################################################################################

# Set current working directory to the directory of the script
cd "$(dirname "$0")"

while [ "$1" != "" ]
do
    case $1 in
        -c) if [ $# -lt 2 ]
            then
                echo "Too few arguments."
                echo ""

                help $0
                exit 1
            else
                shift
                CLUSTERING=$1
                shift
            fi
            ;;

        -h) help $0
            exit 0
            ;;

        -k) if [ $# -lt 2 ]
            then
                echo "Too few arguments."
                echo ""

                help $0
                exit 1
            else
                shift
                K=$1
                shift
            fi
            ;;

        -m) if [ $# -lt 2 ]
            then
                echo "Too few arguments."
                echo ""

                help $0
                exit 1
            else
                shift
                MODEL=$1
                shift
            fi
            ;;

        -n) if [ $# -lt 2 ]
            then
                echo "Too few arguments."
                echo ""

                help $0
                exit 1
            else
                shift
                REPETITIONS=$1
                shift
            fi
            ;;

        -s) if [ $# -lt 2 ]
            then
                echo "Too few arguments."
                echo ""

                help $0
                exit 1
            else
                shift
                SIZE=$1
                shift
            fi
            ;;

        -t) TEST=1
            shift
            ;;

        *)  help $0
            exit 1
            ;;
    esac
done

# Clustering method was not specified
if [ ${CLUSTERING} = "" ] || [ ${MODEL} = "" ]
then
    help $0
    exit 1
fi

# Run clustering several times
for ((i = 0; i <= $REPETITIONS; i++))
do
    case ${CLUSTERING} in
        r)  random ${i}
            ;;

        b)  bgmmm ${i}
            ;;

        c)  crp ${i}
            ;;

        d)  ddcrp ${i}
            ;;

        *)  help $0
            exit 1
            ;;
    esac
done

# Summarize evaluation
./run.sh clustering_system/summary.py `seq 0 ${REPETITIONS}`