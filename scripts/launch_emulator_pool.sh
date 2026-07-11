#!/bin/bash
# Launches N parallel AndroidWorld emulator containers (DMS reproduction
# project eval harness), each on its own console/grpc port pair, all
# sharing the SAME `--network host` namespace as the original
# `dms_emulator` container from scripts/setup_androidworld.sh.
#
# Container `dms_emulator` (instance 0, ports 5554/8554) must already be
# running (created by scripts/setup_androidworld.sh) -- this script only
# launches the ADDITIONAL instances 1..N-1 and performs first-time app
# setup on each of them (installs the same 3rd-party APKs AndroidWorld
# tasks need; cached after the first instance so subsequent instances are
# fast, no re-download).
#
# Usage:
#   scripts/launch_emulator_pool.sh [N]   # N = total instances, default 4
#
# Port scheme for instance i (i=0 is the pre-existing dms_emulator):
#   console_port = 5554 + 2*i   (matches AndroidWorld's own "5554, 5556, ..."
#                                 convention for multiple devices)
#   grpc_port    = 8554 + i
#   container    = dms_emulator            (i=0, pre-existing)
#                  dms_emulator_<i>        (i>=1)
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
ADB="$PROJECT_ROOT/android_sdk_host/platform-tools/adb"

N="${1:-4}"

if ! docker ps --format '{{.Names}}' | grep -qx dms_emulator; then
  echo "ERROR: base instance 'dms_emulator' is not running. Run" >&2
  echo "  scripts/setup_androidworld.sh" >&2
  echo "first to create instance 0 (ports 5554/8554)." >&2
  exit 1
fi

echo "== [1/3] Launching additional emulator containers (instances 1..$((N-1))) =="
for i in $(seq 1 $((N - 1))); do
  name="dms_emulator_${i}"
  console_port=$((5554 + 2 * i))
  grpc_port=$((8554 + i))
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    echo "  $name already exists, skipping launch (docker rm -f $name first to recreate)."
    continue
  fi
  echo "  Launching $name (console_port=$console_port, grpc_port=$grpc_port) ..."
  docker run -d --name "$name" \
    --device /dev/kvm \
    --network host \
    --restart unless-stopped \
    -e CONSOLE_PORT="$console_port" \
    -e GRPC_PORT="$grpc_port" \
    dms_android_emulator:latest >/dev/null
done

echo "== [2/3] Waiting for each new instance to report 'Emulator is ready' (up to 4 min each) =="
for i in $(seq 1 $((N - 1))); do
  name="dms_emulator_${i}"
  for _ in $(seq 1 48); do
    if docker logs "$name" 2>&1 | grep -q "Emulator is ready"; then
      echo "  $name booted."
      break
    fi
    sleep 5
  done
done

echo "== [3/3] First-time app setup (installs 3rd-party APKs; cached after instance 1) =="
source "$PROJECT_ROOT/.venv/bin/activate"
python - "$N" <<'PYEOF'
import sys
from android_world.env import env_launcher

n = int(sys.argv[1])
for i in range(1, n):
  console_port = 5554 + 2 * i
  grpc_port = 8554 + i
  print(f"  Setting up apps on instance {i} (console_port={console_port}) ...")
  env = env_launcher.load_and_setup_env(
      console_port=console_port,
      emulator_setup=True,
      freeze_datetime=True,
      adb_path="android_sdk_host/platform-tools/adb",
      grpc_port=grpc_port,
  )
  state = env.get_state(wait_to_stabilize=False)
  print(f"    OK: screenshot={state.pixels.shape} ui_elements={len(state.ui_elements)}")
  env.close()
PYEOF

echo "All $N emulator instances are ready:"
"$ADB" devices -l
