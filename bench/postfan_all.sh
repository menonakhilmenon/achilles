#!/usr/bin/env bash
cd /var/home/akhil/achilles
R=bench/results/postfan.txt
: > "$R"
echo "== sustained ssd (120s, max fans) ==" >> "$R"
bench/ssdread read bench/scratch.bin -b 20 -t 4 -T 120 -m rand >> "$R" 2>&1
echo "temp after: $(cat /sys/class/nvme/nvme0/hwmon*/temp1_input | head -1)" >> "$R"
bench/ab_bias.sh
cat bench/results/ab-bias-janitor.txt >> "$R"
bench/prefill_ab.sh
cat bench/results/prefill-ab.txt >> "$R"
echo "ALL_DONE" >> "$R"
