use std::path::{Path, PathBuf};
use std::process::Command;

/// Capture the git commit at build time and expose it to the crate via
/// `env!("GIT_COMMIT_HASH")`.
///
/// This records the version of the code that was actually compiled, which is
/// what matters for reproducibility.
///
/// Getting the *re-run* condition right is the whole difficulty. `.git/HEAD`
/// only changes on checkout/branch switch — committing on the current branch
/// rewrites `.git/refs/heads/<branch>` (or `.git/packed-refs`) and leaves
/// `.git/HEAD` untouched. A build script that watches `.git/HEAD` alone is
/// therefore never re-run on a normal commit and freezes the stamp at
/// whatever commit the crate was *first* built at, silently, per build
/// directory. Watch the resolved ref as well.
fn main() {
    let git_dir = run_git(&["rev-parse", "--git-dir"])
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".git"));
    // `--git-common-dir` keeps worktrees pointing at the shared refs.
    let common_dir = run_git(&["rev-parse", "--git-common-dir"])
        .map(PathBuf::from)
        .unwrap_or_else(|| git_dir.clone());

    let commit_hash = run_git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());

    // Uncommitted changes mean no commit describes this binary: say so in the
    // stamp rather than pointing at a commit whose source was not compiled.
    let dirty = run_git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    let stamp = if dirty {
        format!("{commit_hash}-dirty")
    } else {
        commit_hash
    };

    println!("cargo:rustc-env=GIT_COMMIT_HASH={stamp}");

    // Re-run when HEAD moves (checkout / branch switch) ...
    rerun_if_exists(&git_dir.join("HEAD"));
    // ... and when the branch HEAD points at moves (a plain commit).
    if let Some(head_ref) = run_git(&["symbolic-ref", "--quiet", "HEAD"]) {
        rerun_if_exists(&common_dir.join(&head_ref));
    }
    // Refs may live in packed-refs instead of as loose files.
    rerun_if_exists(&common_dir.join("packed-refs"));
    // Staging/worktree changes flip the dirty flag without moving any ref.
    rerun_if_exists(&git_dir.join("index"));
}

/// Emit a `rerun-if-changed` only for paths that exist: naming a missing path
/// makes cargo re-run the script on *every* build.
fn rerun_if_exists(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn run_git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
