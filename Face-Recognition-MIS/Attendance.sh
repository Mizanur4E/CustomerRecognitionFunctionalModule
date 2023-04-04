#!/usr/bin/env bash
#1 1 * * * /home/aci-mis-ai/Face-Recognition-MIS/Attendance.sh
source /home/aci-mis-ai/Face-Recognition-MIS/venv/bin/activate
cd /home/aci-mis-ai/Face-Recognition-MIS
python main.py --server --threshold=0.68
