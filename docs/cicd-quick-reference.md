# CI/CD Quick Reference Card

## 🚀 Quick Fixes

### Fix Formatting Issues

```bash
# Comment on PR:
/fix-precommit

# Or locally:
pre-commit run --all-files
git add .
git commit -m "Fix formatting"
git push
```

### Add Ready Label
1. Go to your PR on GitHub
2. Click "Labels" on the right
3. Select "ready"

## 📊 Check CI/CD Status

### Web Interface
- Check badges on [README](../README.md)
- View workflow runs: https://github.com/vllm-project/tpu-inference/actions

### Command Line

```bash
# Monitor failures
python scripts/monitor_cicd.py --only-failures

# Get detailed info
python scripts/monitor_cicd.py --detailed

# Using GitHub CLI
gh run list --repo vllm-project/tpu-inference
gh run view <run-id> --log
```

## 🔍 Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Pre-commit fails | "hook(s) made changes" | Comment `/fix-precommit` on PR |
| Ready label check fails | "'ready' label NOT found" | Add "ready" label to PR |
| Tests fail | Test workflow fails | Check logs, fix tests, push |
| Build fails | Build workflow fails | Check logs, fix code, push |

## 📚 Resources

- [Full Documentation](cicd-monitoring.md)
- [Implementation Details](cicd-implementation-summary.md)
- [Pre-commit Config](../.pre-commit-config.yaml)

## 🛠️ Local Setup

```bash
# Install pre-commit hook
pip install pre-commit
pre-commit install

# Now hooks run automatically on commit
git commit -m "Your changes"
```

## 🤖 Automated Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| pre-commit | PR/push | Enforce formatting |
| check_ready_label | PR | Ensure PR has ready label |
| auto-fix-precommit | `/fix-precommit` comment | Auto-fix formatting |
| cicd-health-check | Every 6 hours | Monitor CI/CD health |

## 💡 Pro Tips

1. **Install pre-commit locally** to catch issues before pushing
2. **Use the monitoring script** to check status before meetings
3. **Comment `/fix-precommit`** instead of fixing formatting manually
4. **Check the health check issue** for overall CI/CD status
5. **Read the full docs** for in-depth troubleshooting

## 🆘 Get Help

1. Check [CI/CD Monitoring Guide](cicd-monitoring.md)
2. Search existing issues
3. Ask in developer Slack (#sig-tpu)
4. Create an issue with logs

---

**Remember:** Most CI/CD issues can be auto-fixed! 🎉
