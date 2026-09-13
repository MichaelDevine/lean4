#!/usr/bin/env bash
source ../common.sh

# This test covers `--compact-setup`, which changes only the on-disk encoding
# of a module setup file. The default remains pretty-printed, the compact form
# must decode to the same `ModuleSetup`, and the encoding must not behave as a
# semantic build input: toggling it cannot invalidate an up-to-date module.

SETUP=.lake/build/ir/A.setup.json

# Default build: the setup file is pretty-printed, i.e. spread over lines.
./clean.sh
$LAKE build
test_cmd test -s $SETUP
cp $SETUP pretty.json
test_cmd test "$(wc -l < pretty.json)" -gt 1

# Compact build from clean: the same setup, written on a single line.
./clean.sh
$LAKE build --compact-setup
test_cmd test -s $SETUP
cp $SETUP compact.json
test_cmd test "$(wc -l < compact.json)" -le 1

# The encodings differ on disk but the decoded setups do not.
test_cmd_fails diff -q pretty.json compact.json
$LAKE env lean --run equal.lean pretty.json compact.json

# The encoding is not a semantic build input. Rebuilding an up-to-date module
# with the option flipped must not recompile it, so the setup file it already
# has is left exactly as it was. A completed artifact is not invalidated by a
# formatting preference.
cp $SETUP before-toggle.json
$LAKE build
test_cmd diff -u before-toggle.json $SETUP
$LAKE build --compact-setup
test_cmd diff -u before-toggle.json $SETUP

# And the default is genuinely unchanged: a clean default build reproduces the
# original bytes exactly, not merely a file with the same shape.
./clean.sh
$LAKE build
test_cmd diff -u pretty.json $SETUP

# ... and the same from the pretty side: an up-to-date module is not rewritten
# when the option is turned on either.
$LAKE build --compact-setup
test_cmd diff -u pretty.json $SETUP

# Cleanup
rm -f pretty.json compact.json before-toggle.json
