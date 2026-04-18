# Runbook

Operational procedures for the `ring-seq` crate.

## First-time setup

These steps are performed once, before the first release.

### 1. Create a crates.io account

1. Go to [crates.io](https://crates.io) and log in with your GitHub account.
2. Go to **Account Settings** > **API Tokens**.
3. Click **New Token**.
   - Name: `github-actions`
   - Scopes: `publish-update` (or unrestricted for the very first publish,
     since the crate doesn't exist yet).
4. Copy the token.

### 2. Add the token to GitHub

1. Go to the repository on GitHub > **Settings** > **Secrets and variables** >
   **Actions**.
2. Click **New repository secret**.
   - Name: `CARGO_REGISTRY_TOKEN`
   - Value: paste the token from step 1.

## Publishing a release

### What happens

```
v0.1.0 tag pushed
  |
  +---> CI jobs run in parallel (fmt, clippy, test x3, docs, msrv)
  |       |
  |       v all pass
  |
  +---> Version check: tag v0.1.0 == Cargo.toml 0.1.0?
  |       |
  |       v matches
  |
  +---> cargo publish --> crate goes live on crates.io
  |       |
  |       v success
  |
  +---> gh release create --> GitHub Release with auto-generated notes
```

### Steps

1. **Bump the version** in `Cargo.toml`:

   ```toml
   version = "0.2.0"
   ```

2. **Commit and push** to `main`:

   ```bash
   git add Cargo.toml
   git commit -m "Bump version to 0.2.0"
   git push origin main
   ```

3. **Tag and push** the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

The release workflow (`.github/workflows/release.yml`) takes care of the rest.

### Safeguards

- The CI suite (fmt, clippy, test, docs, MSRV) runs before publishing. A
  failure in any job blocks the release.
- The workflow verifies that the Git tag matches the version in `Cargo.toml`.
  A mismatch fails the pipeline.
- The crates.io token is stored as a GitHub secret and never exposed in logs.

## CI overview

The CI workflow (`.github/workflows/ci.yml`) runs on every push to `main` and
on every pull request.

| Job | What it checks | Runs on |
|---|---|---|
| **Rustfmt** | `cargo fmt -- --check` | Ubuntu |
| **Clippy** | `cargo clippy -- -D warnings` | Ubuntu |
| **Test** | `cargo test` (unit + doc tests) | Ubuntu, macOS, Windows |
| **Docs** | `cargo doc --no-deps` with `-D warnings` | Ubuntu |
| **MSRV** | `cargo check` on Rust 1.63 | Ubuntu |

## Routine maintenance

### Updating the MSRV

1. Change `rust-version` in `Cargo.toml`.
2. Update the toolchain version in `.github/workflows/ci.yml` (the `msrv` job).
3. Update the "Minimum Rust version" line in `README.md`.

### Regenerating docs locally

```bash
cargo doc --no-deps --open
```

### Running the full CI checks locally

```bash
cargo fmt -- --check
cargo clippy -- -D warnings
cargo test
cargo doc --no-deps
```
