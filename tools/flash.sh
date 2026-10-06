#!/bin/zsh
# Back up the Mega's current flash, then upload firmware/xy_tb6560.
#   tools/flash.sh            (macOS; auto-detects /dev/cu.usbmodem*)
#   tools/flash.sh /dev/cu.usbmodemXXXX
set -e
setopt null_glob
cd "$(dirname "$0")/.."  # repo root
ports=(/dev/cu.usbmodem* /dev/cu.usbserial*)
PORT=${1:-$ports[1]}
[[ -z "$PORT" ]] && { echo "Arduino port not found"; exit 1; }
if pgrep -f mask_aligner_optimized_v4.py >/dev/null; then
  echo "The web app is running and holds the port. Stop it first (Ctrl+C in its terminal)."; exit 1
fi
AD=$(ls -d ~/Library/Arduino15/packages/arduino/tools/avrdude/*/ | tail -1)
BACKUP=firmware/board_backup_$(date +%Y-%m-%d_%H%M%S).hex
echo "[1/3] backup current flash -> $BACKUP (about 35 s)"
"$AD/bin/avrdude" -C "$AD/etc/avrdude.conf" -p atmega2560 -c wiring -P "$PORT" -b 115200 -D \
  -U flash:r:$BACKUP:i -q -q
echo "[2/3] compile firmware/xy_tb6560"
arduino-cli compile --fqbn arduino:avr:mega firmware/xy_tb6560 >/dev/null
echo "[3/3] upload to $PORT"
arduino-cli upload --fqbn arduino:avr:mega -p "$PORT" firmware/xy_tb6560 >/dev/null
echo "done. restore the old firmware with:"
echo "  $AD/bin/avrdude -C $AD/etc/avrdude.conf -p atmega2560 -c wiring -P $PORT -b 115200 -D -U flash:w:$BACKUP:i"
