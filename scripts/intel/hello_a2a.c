/* hello_a2a.c — multi-node MPI launch test: rank/host map, MPI_Alltoall, check of the received data.
 * Usage: hello_a2a [bytes_per_pair=8192] [iters=20]
 * Prints one line "HELLO_OK ranks=... nodes=... a2a_us=... GBs_node=... lib=..." on success,
 * "HELLO_FAIL ..." if the received data are wrong. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int r, p, len, i, it;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    char host[MPI_MAX_PROCESSOR_NAME], lib[MPI_MAX_LIBRARY_VERSION_STRING];
    MPI_Get_processor_name(host, &len);
    MPI_Get_library_version(lib, &len);
    for (i = 0; lib[i]; ++i) if (lib[i] == '\n') { lib[i] = 0; break; }
    long bytes = argc > 1 ? atol(argv[1]) : 8192;
    int iters = argc > 2 ? atoi(argv[2]) : 20;
    long n = bytes / 8; if (n < 1) n = 1;

    /* number of nodes = number of distinct host names */
    MPI_Comm node; int nr, lead, nnodes;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, r, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &nr);
    lead = (nr == 0);
    MPI_Allreduce(&lead, &nnodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (lead) { printf("  node leader rank %4d on %s\n", r, host); fflush(stdout); }

    double* sb = malloc(sizeof(double) * n * p);
    double* rb = malloc(sizeof(double) * n * p);
    long j;
    for (i = 0; i < p; ++i) for (j = 0; j < n; ++j) sb[(long)i * n + j] = r * 1e6 + i + j * 1e-3;
    MPI_Alltoall(sb, n, MPI_DOUBLE, rb, n, MPI_DOUBLE, MPI_COMM_WORLD);
    int bad = 0, allbad;
    for (i = 0; i < p; ++i) for (j = 0; j < n; ++j) if (rb[(long)i * n + j] != i * 1e6 + r + j * 1e-3) bad = 1;
    MPI_Allreduce(&bad, &allbad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (it = 0; it < iters; ++it) MPI_Alltoall(sb, n, MPI_DOUBLE, rb, n, MPI_DOUBLE, MPI_COMM_WORLD);
    double t = (MPI_Wtime() - t0) / iters, tm;
    MPI_Allreduce(&t, &tm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (r == 0) {
        /* data leaving one node per alltoall: ranks_on_node * (ranks off node) * bytes */
        double rpn = (double)p / nnodes, off = (double)p - rpn;
        double gbs = nnodes > 1 ? rpn * off * n * 8.0 / tm / 1e9 : 0.0;
        printf("%s ranks=%d nodes=%d bytes_per_pair=%ld a2a_us=%.1f GBs_node_offnode=%.2f lib=\"%s\"\n",
               allbad ? "HELLO_FAIL(data)" : "HELLO_OK", p, nnodes, n * 8, tm * 1e6, gbs, lib);
        fflush(stdout);
    }
    free(sb); free(rb);
    MPI_Comm_free(&node);
    MPI_Finalize();
    return allbad;
}
