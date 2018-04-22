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

K=20
ALPHA=0.01
CLUSTERING=""
MODEL=""
ITERS=50
REPETITIONS=5
SIZE=100

################################################################################
# FUNCTIONS                                                                    #
################################################################################

function random {
    args=()
    args+=( "-c news" )
    args+=( "-m random" )
    args+=( "-l dummy" )
    args+=( "-i $1" )
    args+=( "-K ${K}" )
    args+=( "-n ${ITERS}" )
    args+=( "-s ${SIZE}" )
    (( TEST == 1 )) && args+=( '-t' )
    ./run.sh clustering_system/main.py ${args[@]}
}

function bgmmm {
    args=()
    args+=( "-c news" )
    args+=( "-m ${MODEL}" )
    args+=( "-l BGMM" )
    args+=( "-a ${ALPHA}" )
    args+=( "-i $1" )
    args+=( "-K ${K}" )
    args+=( "-n ${ITERS}" )
    args+=( "-s ${SIZE}" )
    (( TEST == 1 )) && args+=( '-t' )
    ./run.sh clustering_system/main.py ${args[@]}
}

function crp {
    args=()
    args+=( "-c news" )
    args+=( "-m ${MODEL}" )
    args+=( "-l CRP" )
    args+=( "-a ${ALPHA}" )
    args+=( "-i $1" )
    args+=( "-K ${K}" )
    args+=( "-n ${ITERS}" )
    args+=( "-s ${SIZE}" )
    (( TEST == 1 )) && args+=( '-t' )
    ./run.sh clustering_system/main.py ${args[@]}
}

function ddcrp {
    args=()
    args+=( "-c news" )
    args+=( "-m ${MODEL}" )
    args+=( "-l ddCRP" )
    args+=( "-a ${ALPHA}" )
    args+=( "-i $1" )
    args+=( "-K ${K}" )
    args+=( "-n ${ITERS}" )
    args+=( "-s ${SIZE}" )
    (( TEST == 1 )) && args+=( '-t' )
    ./run.sh clustering_system/main.py ${args[@]}
}

function help {
    echo "Run clustering experiments"
    echo "--------------------------"
    echo "Copyright (c) Martin Váňa, 2018"
    echo ""
    echo "Usage: $1 [OPTIONS]"
    echo "Available options"
    echo "  -a <alpha>                           the alpha hyperparameter"
    echo "  -c <method={r|b|c|d}>                use clustering method where:"
    echo "                                           r    random"
    echo "                                           b    BGMM"
    echo "                                           c    CRP"
    echo "                                           d    ddCRP"
    echo "  -h                                   displays help"
    echo "  -k <integer>                         the number of clusters (if applicable)"
    echo "  -m <model={random|LSA|LDA|doc2vec}>  use specified model"
    echo "  -n <integer>                         the number of sampling iterations"
    echo "  -r <integer>                         the number of repetitions"
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
        -a) if [ $# -lt 2 ]
            then
                echo "Too few arguments."
                echo ""

                help $0
                exit 1
            else
                shift
                ALPHA=$1
                shift
            fi
            ;;

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
                ITERS=$1
                shift
            fi
            ;;

        -r) if [ $# -lt 2 ]
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