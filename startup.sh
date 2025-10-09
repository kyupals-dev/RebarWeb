#!/bin/bash
# startup.sh — runs on desktop login

# Wait 10 seconds for desktop & Wi-Fi to stabilize
sleep 10

# Directory of your project
WORKDIR="/home/team10/RebarWeb"

# Activate virtual environment and run both scripts in separate terminals
lxterminal --title="Force SSL" --command="bash -c 'cd $WORKDIR && source venv/bin/activate && python3 force_ssl_regen.py; exec bash'" &

# Wait a bit longer to ensure the SSL process starts fully
sleep 5

lxterminal --title="Rebar Main" --command="bash -c 'cd $WORKDIR && source venv/bin/activate && python3 main.py; exec bash'" &
