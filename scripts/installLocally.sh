# Build the ZIP
./release.sh

# Install to the local BEAST package directory
PKG=AdaptiveOperators
BEAST_PKG_DIR=~/.beast/2.8/$PKG

mkdir -p "$BEAST_PKG_DIR"
unzip -o "$PKG.v1.0.0.zip" -d "$BEAST_PKG_DIR"
