#!/bin/bash
# startup.sh for Raspberry Pi 5 - Fullscreen terminals with panel toggle
# Press Ctrl+Alt+P to show/hide the panel

# Wait 10 seconds for desktop & Wi-Fi to stabilize
sleep 10

# Directory of your project
WORKDIR="/home/team10/RebarWeb"

# Create a script to toggle panel visibility
cat > /tmp/toggle_panel.sh << 'EOF'
#!/bin/bash
if pgrep -f wf-panel-pi > /dev/null; then
    # Panel is running, kill it
    pkill -f wf-panel-pi
    notify-send "Panel Hidden" "Press Ctrl+Alt+P to show"
else
    # Panel is not running, start it
    wf-panel-pi &
    notify-send "Panel Shown" "Press Ctrl+Alt+P to hide"
fi
EOF

chmod +x /tmp/toggle_panel.sh

# Bind Ctrl+Alt+P to toggle panel (using xbindkeys if available)
if command -v xbindkeys &> /dev/null; then
    # Create xbindkeys config
    cat > ~/.xbindkeysrc << 'EOF'
# Toggle panel with Ctrl+Alt+P
"bash /tmp/toggle_panel.sh"
    Control+Alt + p
EOF
    killall xbindkeys 2>/dev/null
    xbindkeys
fi

# Hide the panel initially
pkill -f wf-panel-pi

# Run Force SSL in fullscreen terminal
lxterminal --title="Force SSL" --geometry=200x50 --command="bash -c 'cd $WORKDIR && source venv/bin/activate && python3 force_ssl_regen.py; exec bash'" &

# Get the window ID of the last opened terminal and maximize it
sleep 1
wmctrl -r "Force SSL" -b add,maximized_vert,maximized_horz

# Wait a bit longer to ensure the SSL process starts fully
sleep 5

# Run Rebar Main in fullscreen terminal
lxterminal --title="Rebar Main" --geometry=200x50 --command="bash -c 'cd $WORKDIR && source venv/bin/activate && python3 main.py; exec bash'" &

# Maximize the Rebar Main terminal
sleep 1
wmctrl -r "Rebar Main" -b add,maximized_vert,maximized_horz

# Show notification about keyboard shortcut
sleep 2
notify-send "Rebar Vista Started" "Press Ctrl+Alt+P to toggle panel visibility" -t 5000

# Script finished - no infinite loop
exit 0