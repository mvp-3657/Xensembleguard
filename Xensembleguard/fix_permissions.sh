#!/bin/bash
echo "Granting packet capture permissions..."
sudo chmod o+r /dev/bpf*
echo "✅ Done! Now run: python live_capture_dashboard_v2.py"
