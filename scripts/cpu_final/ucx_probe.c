/* ucx_probe.c — LD_PRELOAD library: MPI rank 0 prints which UCX the process really loaded; the solvers are not changed.
 *     UCXPROBE version=<x.y.z|none|unknown> lib=<path of libucp|-> uct=<path of libuct|-> at=<finalize|exit> [reason=...]
 * It wraps MPI_Finalize (PMPI): at that point MPI_Init has loaded libfabric/UCX and they are still mapped, and the
 * rank is exact (MPI_Comm_rank). If MPI_Finalize is never reached, a destructor reports at process exit instead
 * (rank from PMI_RANK, PMIX_RANK, OMPI_COMM_WORLD_RANK or SLURM_PROCID).
 *   version=none     no libucp mapped (single node with I_MPI_FABRICS=shm, or a provider without UCX)
 *   version=unknown  libucp mapped but its version could not be read; reason= says why (dlopen / dlsym error)
 * Build with the MPI of the container (mpi.h, PMPI_Finalize):  mpicc -O2 -shared -fPIC ucx_probe.c -o ucx_probe.so -ldl */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int reported = 0;

static int env_rank0(void) {
    const char* names[] = {"PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID", 0};
    const char* r = 0;
    for (int i = 0; names[i] && !(r && *r); ++i) r = getenv(names[i]);
    return r && !strcmp(r, "0");
}

static void mapped(const char* key, char* out, size_t n) {
    FILE* f = fopen("/proc/self/maps", "r");
    char line[4096];
    snprintf(out, n, "-");
    while (f && fgets(line, sizeof line, f)) {
        char* p = strchr(line, '/');
        if (!p || !strstr(p, key)) continue;
        p[strcspn(p, "\n")] = 0;
        snprintf(out, n, "%s", p);
        break;
    }
    if (f) fclose(f);
}

static void report(const char* at) {
    char ucp[1024], uct[1024], ver[64] = "none", why[512] = "";
    reported = 1;
    mapped("/libucp.so", ucp, sizeof ucp);
    mapped("/libuct.so", uct, sizeof uct);
    if (!strcmp(ucp, "-")) {
        snprintf(why, sizeof why, " reason=no-libucp-mapped");
    } else {
        void* h = dlopen(ucp, RTLD_NOW | RTLD_NOLOAD);
        const char* e1 = h ? 0 : dlerror();
        if (!h) h = dlopen("libucp.so.0", RTLD_NOW | RTLD_NOLOAD);
        void (*gv)(unsigned*, unsigned*, unsigned*) = 0;
        if (h) gv = (void (*)(unsigned*, unsigned*, unsigned*))dlsym(h, "ucp_get_version");
        if (gv) {
            unsigned a, b, c; gv(&a, &b, &c); snprintf(ver, sizeof ver, "%u.%u.%u", a, b, c);
        } else {
            snprintf(ver, sizeof ver, "unknown");
            const char* e2 = dlerror();
            snprintf(why, sizeof why, " reason=%s:%s", h ? "dlsym-ucp_get_version" : "dlopen-NOLOAD",
                     e2 ? e2 : (e1 ? e1 : "?"));
            for (char* q = why; *q; ++q) if (*q == ' ' && q != why) *q = '_';
        }
    }
    fprintf(stdout, "UCXPROBE version=%s lib=%s uct=%s at=%s%s\n", ver, ucp, uct, at, why);
    fflush(stdout);
}

int MPI_Finalize(void) {
    int r = -1, init = 0, fin = 0;
    MPI_Initialized(&init); MPI_Finalized(&fin);
    if (init && !fin) MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if (r == 0) report("finalize");
    reported = 1;                      /* other ranks: no report at exit either */
    return PMPI_Finalize();
}

__attribute__((destructor)) static void ucx_probe_exit(void) {
    if (!reported && env_rank0()) report("exit");
}
