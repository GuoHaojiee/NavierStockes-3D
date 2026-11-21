#!/bin/bash
# rebuild_inside.sh — runs AS ROOT inside a writable copy of ns3d_intel.sqsh (started by rebuild_container.sh).
# Adds the user-space InfiniBand stack, UCX and the PMI-2 / PMIx client libraries with apt (Ubuntu 22.04),
# then writes /opt/ns3d_pmi_libs.txt (paths used by hello_2node.sbatch) and prints what is where.
# Exit code != 0 if apt fails or a library that must exist is missing.
export DEBIAN_FRONTEND=noninteractive
. /etc/os-release; echo "image: $PRETTY_NAME ($VERSION_CODENAME), $(uname -m)"
PKGS=${RC_PKGS:-"rdma-core libibverbs1 ibverbs-providers ibverbs-utils librdmacm1 libucx0 ucx-utils libpmi2-0 libpmix2"}

apt-get update -qq || { echo "FAIL apt-get update (no network on this node? proxy?)"; exit 1; }
have=""; skipped=""
for p in $PKGS; do
    v=$(apt-cache policy $p 2>/dev/null | awk '/Candidate:/ {print $2}')
    if [ -n "$v" ] && [ "$v" != "(none)" ]; then have="$have $p"; printf "  %-18s %s\n" $p "$v"
    else skipped="$skipped $p"; echo "SKIP $p: no such package in Ubuntu $VERSION_CODENAME (enabled components: $(grep -hE '^deb ' /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null | awk '{for (i = 4; i <= NF; i++) print $i}' | sort -u | tr '\n' ' '))"; fi
done
echo "apt-get install$have"
apt-get install -y -qq --no-install-recommends $have || { echo "FAIL apt-get install"; exit 1; }
ldconfig
apt-get clean; rm -rf /var/lib/apt/lists/*

echo "--- libraries"
st=0
for l in libpmi2.so libpmix.so libibverbs.so libucp.so; do
    f=$(ldconfig -p | awk -v l=$l 'index($1, l) == 1 {print $NF}' | sort -u | tr '\n' ' ')
    [ -z "$f" ] && f=$(find /usr/lib /usr/lib64 /lib -name "$l*" 2>/dev/null | sort -u | tr '\n' ' ')
    if [ -n "$f" ]; then printf "OK   %-14s %s\n" $l "$f"; else printf "FAIL %-14s not found\n" $l; st=1; fi
done
echo "--- ibverbs providers (rdma-core): $(ls /usr/lib/x86_64-linux-gnu/libibverbs/ 2>/dev/null | tr '\n' ' ')"
command -v ucx_info >/dev/null && echo "--- $(ucx_info -v 2>/dev/null | head -2 | tr '\n' ' ')"

# paths for hello_2node.sbatch (D variants): the versioned PMI-2 client and the libpmix.so.2 soname link
PMI2_LIB=$(ls /usr/lib/x86_64-linux-gnu/libpmi2.so.0 /usr/lib/x86_64-linux-gnu/libpmi2.so 2>/dev/null | head -1)
PMIX_LIB=$(ls /usr/lib/x86_64-linux-gnu/libpmix.so.2 /usr/lib/x86_64-linux-gnu/pmix2/lib/libpmix.so.2 2>/dev/null | head -1)
{ echo "PMI2_LIB=$PMI2_LIB"; echo "PMIX_LIB=$PMIX_LIB"; echo "SKIPPED_PKGS=\"${skipped# }\""; echo "BUILT=\"$(date '+%F %T')\""; } > /opt/ns3d_pmi_libs.txt
echo "--- /opt/ns3d_pmi_libs.txt"; cat /opt/ns3d_pmi_libs.txt
[ -n "$PMI2_LIB" ] && [ -n "$PMIX_LIB" ] || st=1

if [ -f /etc/profile.d/ns3d_intel.sh ]; then echo "OK   /etc/profile.d/ns3d_intel.sh still there"; else echo "FAIL /etc/profile.d/ns3d_intel.sh missing"; st=1; fi
[ -n "$skipped" ] && echo "skipped packages:$skipped"
exit $st
