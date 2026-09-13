#!/usr/bin/env python3
# Copyright (c) 2026 Michael Devine. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.
"""Additive Git revision controls; existing upstream test registration is unchanged."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def run(module):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    with tempfile.TemporaryDirectory(prefix="lean git revision ") as temporary:
        root = Path(temporary)
        repo = root / "repository"
        def command(args, cwd=repo):
            return subprocess.check_output([str(arg) for arg in args], cwd=cwd, env=env,
                                           stderr=subprocess.STDOUT, text=True).strip()
        command(["git", "init", "-b", "main", repo], cwd=root)
        command(["git", "config", "user.name", "Lean revision test"])
        command(["git", "config", "user.email", "lean-test@example.invalid"])
        (repo / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.21)
project(RevisionControl NONE)
find_package(Git REQUIRED)
include("${REVISION_MODULE}")
get_git_head_revision(HEAD_REF HEAD_HASH)
file(WRITE "${CMAKE_BINARY_DIR}/revision.txt" "${HEAD_HASH}\\n")
''')
        command(["git", "add", "--", "CMakeLists.txt"])
        command(["git", "commit", "-m", "Initial revision fixture\n\nModel: GPT-5 Codex"])
        first = command(["git", "rev-parse", "HEAD"])
        command(["git", "commit", "--allow-empty", "-m", "Second revision fixture\n\nModel: GPT-5 Codex"])
        second = command(["git", "rev-parse", "HEAD"])
        assert first != second
        def check(label, source, expected):
            build = root / (label + " build")
            command(["cmake", "-S", source, "-B", build, "-DREVISION_MODULE=" + str(module)], cwd=root)
            actual = (build / "revision.txt").read_text().strip()
            assert actual == expected, (label, actual, expected)
            print("git_revision_pass", label, flush=True)
        check("ordinary-attached", repo, second)
        detached = root / "detached worktree"
        command(["git", "worktree", "add", "--detach", detached, first])
        # The worktree HEAD differs from both the common HEAD ref text and its
        # resolved commit. Either incorrect common-HEAD interpretation must fail.
        check("worktree-detached", detached, first)
        attached = root / "attached worktree"
        command(["git", "worktree", "add", "-b", "fixture-branch", attached, first])
        check("worktree-attached", attached, first)
        command(["git", "pack-refs", "--all"])
        check("worktree-attached-packed", attached, first)
        command(["git", "checkout", "--detach", second])
        check("ordinary-detached", repo, second)
        check("worktree-detached-common-detached", detached, first)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, default=Path(__file__).resolve().parents[2] /
                        "src/cmake/Modules/GetGitRevisionDescription.cmake")
    run(parser.parse_args().module.resolve(strict=True))
