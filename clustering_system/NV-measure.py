import numpy as np
import matplotlib.pyplot as plt

from evaluator.measures import v_measure, nv_measure, nv_measure

if __name__ == "__main__":
    K = 10      # the number of clusters
    N_k = 6  # the number of observations in cluster
    p = 1      # metric parameter

    classes = np.array([x for x in range(K) for y in range(N_k)])
    # clusters = np.array([x for x in range(K) for y in range(N_k)])

    clusters = np.array([x for x in range(K * N_k)])

    # clusters = np.copy(classes)
    # np.random.shuffle(clusters)
    # clusters[clusters == 0] = 1 # remove one cluster

    # clusters = np.array([x for x in range(K // 2) for y in range(N_k * 2)])

    print("K:           %d" % len(np.unique(clusters)))
    print("C:           %d" % len(np.unique(classes)))
    print("V-measure:   %f" % v_measure(clusters, classes))
    print("NV-measure:  %f" % nv_measure(clusters, classes, p=p))
    print("NV2-measure: %f" % nv_measure(clusters, classes, p=p))

    ps = [0.5, 1, 1.5, 2, 3, 6]
    x = np.arange(0., 1.001, 0.001)

    fig = plt.figure()
    plt.axis('equal')

    for p in ps:
        plt.plot(x, 1 - (1 - x**p)**(1 / p), label="%s" % str(p))

    plt.title("1 - (1 - x^p)^(1 / p)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

    # fig.savefig("nv-norm.pdf")
