#!/bin/bash
echo "=========================================="
echo " One-Time Permission Fix for Packet Capture"
echo "=========================================="
echo ""
echo "Granting access to network device..."
sudo chmod o+r /dev/bpf*
echo ""
echo "✅ Done! Now run your dashboard:"
echo "   streamlit run live_capture_dashboard.py --server.port 8503"
