#!/usr/bin/env bash
# Staged-envelope GLM-5.2 run: load under a strict ceiling (the loader's page
# cache flood provably reclaims), then raise it for decode once the arena has
# installed and swept. Single short run; results + peaks to stdout.
set -u
cd "$(dirname "$0")/.."
. bench/vkdev.sh
M=models/glm52-gguf/UD-Q2_K_XL/GLM-5.2-UD-Q2_K_XL-00001-of-00007.gguf
LOG=/tmp/claude-1000/staged.log
UNIT=achilles-staged

AVAIL=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if [ "$AVAIL" -lt 48 ]; then
  echo "PREFLIGHT ABORT: only ${AVAIL}GB available (need 48+; leaked kernel memory? reboot)"
  exit 1
fi
systemctl --user reset-failed $UNIT.service 2>/dev/null
rm -f /tmp/claude-1000/gate
systemd-run --user --unit $UNIT -p WorkingDirectory="$PWD" -p MemoryHigh=46G -p MemoryMax=48G -p MemorySwapMax=2G \
  bash -c "GGML_VK_VISIBLE_DEVICES=$VKDEV exec src/achilles-arena -m '$M' \
    -p 'Explain how mixture-of-experts language models work, covering routing, expert specialization, and why sparsity helps.' \
    -n 24 -t 10 -ngl 99 -ot exps=CPU --budget-gib 32 -ub 2048 --workers 6 --pstream 1 --policy reuse --stats > $LOG 2>&1"

# wait for the arena install marker, then raise the ceiling for decode
for i in $(seq 1 120); do
  grep -q "replaced .* GiB with anonymous" $LOG 2>/dev/null && break
  systemctl --user is-active --quiet $UNIT.service || break
  sleep 2
done
CG=$(systemctl --user show $UNIT.service -p ControlGroup --value)
touch /tmp/claude-1000/gate   # single flat ceiling; no staging needed
echo "flat ceiling: memory.high=$(cat /sys/fs/cgroup$CG/memory.high 2>/dev/null)"
( while systemctl --user is-active --quiet $UNIT.service; do
    echo "high=$(cat /sys/fs/cgroup$CG/memory.high 2>/dev/null) cur=$(( $(cat /sys/fs/cgroup$CG/memory.current 2>/dev/null || echo 0) / 2**30 ))G"
    sleep 5
  done ) &
MON=$!

# wait for completion
for i in $(seq 1 150); do
  systemctl --user is-active --quiet $UNIT.service || break
  sleep 2
done
kill $MON 2>/dev/null
echo "=== result ==="
grep -aE "ACHILLES|OUTPUT|replaced" $LOG
journalctl --user -u $UNIT.service -n 4 --no-pager | grep -E "Consumed|Failed" | tail -2
