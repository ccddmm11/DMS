#!/bin/bash
# Build and launch the AndroidWorld headless emulator in Docker (KVM-accelerated),
# then verify host-side adb / AndroidWorld Python connectivity.
#
# Background: this host's Docker daemon has a broken registry proxy
# (/etc/docker/daemon.json + systemd drop-in point at dead proxy ports), so the
# base image `eclipse-temurin:18-jdk` must be pre-fetched with skopeo (which
# respects the shell's own working http_proxy) and loaded with `docker load`
# before building. Also: adbd/console inside the emulator only bind to
# 127.0.0.1, so `-p host:container` port publishing produces an "offline"
# device over NAT hairpin; `--network host` avoids NAT entirely and is required.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

echo "== [1/5] Ensure host adb (platform-tools) is present =="
if [ ! -x "$PROJECT_ROOT/android_sdk_host/platform-tools/adb" ]; then
  mkdir -p "$PROJECT_ROOT/android_sdk_host"
  curl -sL -o /tmp/platform-tools.zip https://dl.google.com/android/repository/platform-tools-latest-linux.zip
  unzip -q -o /tmp/platform-tools.zip -d "$PROJECT_ROOT/android_sdk_host"
fi
ADB="$PROJECT_ROOT/android_sdk_host/platform-tools/adb"
"$ADB" version

echo "== [2/5] Ensure base image eclipse-temurin:18-jdk is loaded locally =="
if ! docker image inspect eclipse-temurin:18-jdk >/dev/null 2>&1; then
  skopeo copy docker://eclipse-temurin:18-jdk docker-archive:/tmp/temurin18.tar:eclipse-temurin:18-jdk
  docker load -i /tmp/temurin18.tar
fi

echo "== [3/5] Build dms_android_emulator image (skip if already built) =="
if ! docker image inspect dms_android_emulator:latest >/dev/null 2>&1; then
  DOCKER_BUILDKIT=0 docker build --network=host \
    --build-arg HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7894}" \
    --build-arg HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7894}" \
    -f "$PROJECT_ROOT/scripts/Dockerfile.emulator" \
    -t dms_android_emulator:latest \
    "$PROJECT_ROOT/android_world"
fi

echo "== [4/5] (Re)launch emulator container with --network host + KVM =="
docker rm -f dms_emulator >/dev/null 2>&1 || true
docker run -d --name dms_emulator \
  --device /dev/kvm \
  --network host \
  --restart unless-stopped \
  dms_android_emulator:latest

echo "Waiting for emulator to boot (up to 4 min)..."
for i in $(seq 1 48); do
  if docker logs dms_emulator 2>&1 | grep -q "Emulator is ready"; then
    echo "Emulator booted after ~$((i*5))s"
    break
  fi
  sleep 5
done

echo "== [5/5] Verify host adb + AndroidWorld Python connectivity =="
"$ADB" disconnect >/dev/null 2>&1 || true
"$ADB" devices -l
"$ADB" -s emulator-5554 shell getprop sys.boot_completed

source "$PROJECT_ROOT/.venv/bin/activate"
python -c "
from android_world.env import env_launcher
env = env_launcher.load_and_setup_env(
    console_port=5554,
    emulator_setup=False,
    freeze_datetime=True,
    adb_path='$ADB',
    grpc_port=8554,
)
state = env.get_state(wait_to_stabilize=False)
print('OK: screenshot', state.pixels.shape, 'ui_elements', len(state.ui_elements))
env.close()
"
echo "AndroidWorld environment is ready."
