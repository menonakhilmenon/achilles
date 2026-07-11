#!/usr/bin/env bash
# Full NVMe read sweep + RAM/SSD interference test. Run on a quiet system.
set -euo pipefail
cd "$(dirname "$0")"
OUT=results/ssd-sweep.jsonl
: > "$OUT"

echo "# random reads: block size x threads" | tee -a "$OUT"
for bs in 1 4 10 20 40; do
    for t in 1 2 4 8 16; do
        ./ssdread read scratch.bin -b $bs -t $t -T 8 -m rand | tee -a "$OUT"
    done
done

echo "# sequential reads" | tee -a "$OUT"
for bs in 4 20; do
    for t in 1 4 8; do
        ./ssdread read scratch.bin -b $bs -t $t -T 8 -m seq | tee -a "$OUT"
    done
done

echo "# interference: 20MiB rand t8 SSD while membw(8t) runs" | tee -a "$OUT"
./ssdread read scratch.bin -b 20 -t 8 -T 25 -m rand > /tmp/ssd_if.json &
SSD_PID=$!
sleep 2
./membw -t 8 -g 6 -r 5 > /tmp/mem_if.json
wait $SSD_PID
{ printf '{"interference":{"ssd":'; cat /tmp/ssd_if.json | tr -d '\n'
  printf ',"membw":'; cat /tmp/mem_if.json | tr -d '\n'; printf '}}\n'; } | tee -a "$OUT"
echo "SWEEP_DONE"
