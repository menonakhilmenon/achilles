#!/usr/bin/env bash
cd /var/home/akhil/achilles
R=bench/results/postfan.txt
: > "$R"
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
( while sleep 10; do
    awk -v t=$(date +%s) '$1=="anon"||$1=="file"||$1=="kernel" {printf "%s=%dMB ", $1, $2/1048576} END {print "t="t}' "$CG/memory.stat" >> bench/results/memstat.txt
  done ) &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT
echo "== sustained ssd (120s, max fans) ==" >> "$R"
bench/ssdread read bench/scratch.bin -b 20 -t 4 -T 120 -m rand >> "$R" 2>&1
echo "temp after: $(cat /sys/class/nvme/nvme0/hwmon*/temp1_input | head -1)" >> "$R"
bench/ab_bias.sh
cat bench/results/ab-bias-janitor.txt >> "$R"
bench/prefill_ab.sh
cat bench/results/prefill-ab.txt >> "$R"
echo "ALL_DONE" >> "$R"
