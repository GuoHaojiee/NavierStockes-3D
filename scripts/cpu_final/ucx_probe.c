/* ucx_probe.c — LD_PRELOAD library for scripts/cpu_final: when the solver exits, MPI rank 0 prints which UCX it
 * really loaded (the solvers themselves are not changed):
 *     UCXPROBE version=<x.y.z|none> lib=<path of libucp|-> uct=<path of libuct|->
 * "none": no libucp was mapped (single node with I_MPI_FABRICS=shm does not load libfabric/UCX).
 * Rank: PMI_RANK (Intel MPI hydra / PMI-2) if set, otherwise SLURM_PROCID. Processes without a rank (mpirun,
 * hydra proxies) print nothing — single-node runs pass the library with mpirun -genv, so they never see it.
 * The version comes from ucp_get_version() of the libucp already in the process (dlopen RTLD_NOLOAD).
 * Build:  icx -O2 -shared -fPIC ucx_probe.c -o ucx_probe.so -ldl                                        */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_rank0(void) {
    const char* r = getenv("PMI_RANK");
    if (!r || !*r) r = getenv("SLURM_PROCID");
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

__attribute__((destructor)) static void ucx_probe_report(void) {
    if (!is_rank0()) return;
    char ucp[1024], uct[1024], ver[64] = "none";
    mapped("/libucp.so", ucp, sizeof ucp);
    mapped("/libuct.so", uct, sizeof uct);
    if (strcmp(ucp, "-")) {
        void* h = dlopen(ucp, RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen("libucp.so.0", RTLD_NOW | RTLD_NOLOAD);
        void (*gv)(unsigned*, unsigned*, unsigned*) = h ? (void (*)(unsigned*, unsigned*, unsigned*))dlsym(h, "ucp_get_version") : 0;
        if (gv) { unsigned a, b, c; gv(&a, &b, &c); snprintf(ver, sizeof ver, "%u.%u.%u", a, b, c); }
        else snprintf(ver, sizeof ver, "unknown");
    }
    fprintf(stdout, "UCXPROBE version=%s lib=%s uct=%s\n", ver, ucp, uct);
    fflush(stdout);
}
