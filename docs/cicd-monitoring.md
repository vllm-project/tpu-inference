# CI/CD Monitoring and Troubleshooting Guide

This guide helps you monitor and fix common CI/CD issues in the tpu-inference repository.

## Quick Start

### Monitor CI/CD Status

To check the current CI/CD status and see any recent failures:

```bash
# Using the monitoring script (requires GitHub CLI)
python scripts/monitor_cicd.py

# Show only failures
python scripts/monitor_cicd.py --only-failures

# Get detailed output
python scripts/monitor_cicd.py --detailed

# JSON output for automation
python scripts/monitor_cicd.py --json
```

### Using GitHub CLI Directly

```bash
# List recent workflow runs
gh run list --repo vllm-project/tpu-inference

# View specific run
gh run view <run-id> --repo vllm-project/tpu-inference

# View logs for a failed run
gh run view <run-id> --log --repo vllm-project/tpu-inference
```

## Common CI/CD Issues and Fixes

### 1. Pre-commit Formatting Failures

**Symptoms:**
- Pre-commit workflow fails with formatting errors
- Error message: "pre-commit hook(s) made changes"

**Cause:** Code doesn't conform to the project's formatting standards (isort, yapf, ruff, etc.)

**Fix Options:**

#### Option A: Auto-fix in PR (Recommended)
Comment on your PR:

```
/fix-precommit
```

The auto-fix workflow will automatically format your code and push the changes.

#### Option B: Fix Locally

```bash
# Install pre-commit
pip install pre-commit

# Run pre-commit on all files
pre-commit run --all-files

# Commit the changes
git add .
git commit -m "Fix formatting issues"
git push
```

#### Option C: Install Pre-commit Hook
To prevent formatting issues in future commits:

```bash
# Install the pre-commit hook
pre-commit install

# Now pre-commit will run automatically on every commit
git commit -m "Your commit message"
```

### 2. Ready Label Check Failures

**Symptoms:**
- "Enforce Ready Label" workflow fails
- Error message: "'ready' label NOT found"

**Cause:** PR is missing the required "ready" label

**Fix:**
1. Go to your PR on GitHub
2. Add the "ready" label to the PR
3. The check will automatically re-run and pass

**Note:** This label is required to ensure PRs are explicitly marked as ready for review/merge.

### 3. Test Failures

**Symptoms:**
- Test workflow fails
- Error messages about failing tests

**Fix:**
1. Check the workflow logs to identify which tests failed
2. Run the tests locally:
  
```bash
   # Install test dependencies
   pip install -r requirements_test.txt
   
   # Run specific test
   pytest tests/path/to/test_file.py::test_name
   
   # Run all tests
   pytest tests/
   ```

1. Fix the failing tests
2. Commit and push the fixes

### 4. Build Failures

**Symptoms:**
- Build workflow fails
- Compilation or installation errors

**Fix:**
1. Check the build logs for specific error messages
2. Try to reproduce locally:
  
```bash
   pip install -e .
   ```

1. Fix any dependency or code issues
2. Commit and push the fixes

### 5. Buildkite Pipeline Failures

**Symptoms:**
- Buildkite jobs fail
- Model tests or integration tests fail

**Fix:**
1. Check Buildkite UI for detailed logs
2. Review the specific pipeline configuration in `.buildkite/`
3. Run integration tests locally if possible
4. Fix the issues and push changes

## Workflow Files

The CI/CD workflows are defined in:
- `.github/workflows/` - GitHub Actions workflows
- `.buildkite/` - Buildkite pipeline configurations

### Key Workflows

1. **pre-commit.yml** - Enforces code formatting and linting
2. **check_ready_label.yml** - Ensures PRs are marked as ready
3. **auto-fix-precommit.yml** - Auto-fixes formatting issues on demand
4. **release.yml** - Handles package releases

## Pre-commit Hooks

The repository uses the following pre-commit hooks:

- **addlicense** - Ensures all files have license headers
- **isort** - Sorts Python imports
- **yapf** - Formats Python code
- **ruff** - Lints and fixes Python code
- **clang-format** - Formats C++/CUDA code
- **PyMarkdown** - Formats Markdown files
- **actionlint** - Lints GitHub Actions workflows
- **shellcheck** - Lints shell scripts
- **detect-missing-init** - Ensures `__init__.py` files exist

## Monitoring Best Practices

1. **Regular Monitoring**: Check CI/CD status regularly, especially after merging PRs
2. **Fix Quickly**: Address failures promptly to prevent blocking other work
3. **Prevention**: Install pre-commit hooks locally to catch issues before pushing
4. **Documentation**: Update this guide when new patterns emerge

## Automated Monitoring

You can set up automated monitoring using the monitoring script:

```bash
# Add to your CI/CD pipeline or cron job
python scripts/monitor_cicd.py --only-failures --json > cicd_status.json

# Send alerts based on the output
if [ $? -ne 0 ]; then
  # Send alert (e.g., via Slack, email, etc.)
  echo "CI/CD failures detected!"
fi
```

## Getting Help

If you encounter an issue not covered in this guide:

1. Check the workflow logs for detailed error messages
2. Search for similar issues in the repository
3. Ask in the development chat/forum
4. Create an issue with the workflow logs and error details

## Contributing

If you find ways to improve CI/CD monitoring or fix common issues:

1. Update this documentation
2. Consider adding automated fixes
3. Submit a PR with your improvements
