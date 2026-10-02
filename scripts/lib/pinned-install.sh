# shellcheck shell=bash
# Pinned, checksum-verified installers for provisioning scripts.
# Source this file; never pipe a remote installer into a shell.
#
# Bump a pin by updating the version AND its sha256 together; checksums come from
# the upstream release (`<artifact>.sha256`) and must be verified against a download.

RUSTUP_PIN_VERSION="1.29.1"
RUST_TOOLCHAIN_PIN="${RUST_TOOLCHAIN_PIN:-1.95.0}"
UV_PIN_VERSION="0.12.21"
DOCKER_APT_KEY_FPR="9DC858229FC7DD38854AE2D88D81803C0EBFCD88"

_pinned_triple() {
  case "$(uname -m)" in
    x86_64) echo x86_64-unknown-linux-gnu ;;
    aarch64 | arm64) echo aarch64-unknown-linux-gnu ;;
    *) echo "pinned-install: unsupported arch $(uname -m)" >&2; return 1 ;;
  esac
}

_pinned_fetch() { # url sha256 dest
  curl --proto '=https' --tlsv1.2 -fsSL -o "$3" "$1"
  if ! echo "$2  $3" | sha256sum -c --quiet -; then
    echo "pinned-install: checksum mismatch for $1" >&2
    rm -f "$3"
    return 1
  fi
}

# Installs rustup + a pinned toolchain (minimal profile) into $CARGO_HOME/$RUSTUP_HOME.
install_rust_pinned() {
  local triple sha tmp
  triple="$(_pinned_triple)" || return 1
  case "$triple" in
    x86_64-unknown-linux-gnu) sha=dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71 ;;
    aarch64-unknown-linux-gnu) sha=15f6e4ce9f583b929c996c91562bad6d4454f3281de858b02cdfdef615fac433 ;;
  esac
  tmp="$(mktemp -d)"
  _pinned_fetch "https://static.rust-lang.org/rustup/archive/${RUSTUP_PIN_VERSION}/${triple}/rustup-init" \
    "$sha" "$tmp/rustup-init" || { rm -rf "$tmp"; return 1; }
  chmod +x "$tmp/rustup-init"
  "$tmp/rustup-init" -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN_PIN"
  rm -rf "$tmp"
}

# Installs uv into ~/.local/bin.
install_uv_pinned() {
  local triple sha tmp
  triple="$(_pinned_triple)" || return 1
  case "$triple" in
    x86_64-unknown-linux-gnu) sha=23f02075b652bb1df64178cfae41b5caf160822e720e2663568f3f5d63bc52c0 ;;
    aarch64-unknown-linux-gnu) sha=030b69227b40af8c1981b7301793dc66e71ed3c796ea8688209dd268bd91ec51 ;;
  esac
  tmp="$(mktemp -d)"
  _pinned_fetch "https://github.com/astral-sh/uv/releases/download/${UV_PIN_VERSION}/uv-${triple}.tar.gz" \
    "$sha" "$tmp/uv.tar.gz" || { rm -rf "$tmp"; return 1; }
  tar -xzf "$tmp/uv.tar.gz" -C "$tmp"
  mkdir -p "$HOME/.local/bin"
  install -m 0755 "$tmp/uv-${triple}/uv" "$tmp/uv-${triple}/uvx" "$HOME/.local/bin/"
  rm -rf "$tmp"
}

# Installs Docker Engine from Docker's apt repo (Debian/Ubuntu) with a fingerprint-checked keyring.
install_docker_apt() {
  local distro codename key
  # shellcheck disable=SC1091
  . /etc/os-release
  distro="$ID"
  codename="${VERSION_CODENAME:?pinned-install: VERSION_CODENAME missing}"
  key="$(mktemp)"
  sudo apt-get update -qq
  sudo apt-get install -y -qq ca-certificates curl gnupg
  curl --proto '=https' --tlsv1.2 -fsSL -o "$key" "https://download.docker.com/linux/${distro}/gpg"
  if ! gpg --show-keys --with-colons "$key" | awk -F: '$1=="fpr"{print $10; exit}' | grep -qx "$DOCKER_APT_KEY_FPR"; then
    echo "pinned-install: Docker apt key fingerprint mismatch" >&2
    rm -f "$key"
    return 1
  fi
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg "$key"
  rm -f "$key"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${distro} ${codename} stable" |
    sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io
}
