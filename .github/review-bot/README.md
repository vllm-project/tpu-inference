# Review bot settings and guide

The review bot, `cbs-oss-review-bot`, reviews pull requests in this repo with
Claude. It posts a GitHub review with a summary and inline comments. It never
approves a PR or blocks a merge. Repo owners control it with the files in this
folder.

The bot reads these files from `main`, so an edit takes effect once its PR
merges. A PR that edits them is still reviewed with the settings on `main`.

Each review ends with links to the instructions it followed (`REVIEW.md` and
the code-rule files), at the commit it read them from.

## Files

| File | What it does |
| --- | --- |
| `config.yaml` | Settings, listed below. A missing file or setting means the default. |
| `REVIEW.md` | The review guide: what the bot checks and how it comments. See [REVIEW.md](#reviewmd). |
| `AGENTS.md` | Code rules for people and coding agents, and the bot's code rules by default (see `code_review_instructions`). This repo has none yet. |

Instructions that are only for reviews go in `REVIEW.md`. If the repo adds an
`AGENTS.md`, it wins over `REVIEW.md` where the two disagree about a code rule.

## Settings in config.yaml

The values in `config.yaml` today are the defaults.

| Setting | Possible values | Default | What it does |
| --- | --- | --- | --- |
| `nits` | `summary`, `inline`, `off` | `summary` | Where nits go. A nit is optional polish: naming, wording, a small cleanup. `summary` lists them in the review text, `inline` posts each on its line, `off` drops them. At most 10 per review. |
| `questions` | `summary`, `inline`, `off` | `summary` | Where questions go. A question is something the bot suspects but could not confirm in the code. Same choices as `nits`. At most 5 per review. |
| `max_inline_comments` | Whole number, 0 to 30 | `15` | Most inline comments in one review, counting problems, nits and questions. Problems come first. Problems past the limit go to the summary under "More problems". `0` puts everything in the summary. |
| `inline_labels` | `"blocking, should-fix"` or `blocking` | `"blocking, should-fix"` | Which problems get inline comments. A problem is labeled `blocking` (fix before merging) or `should-fix`. With only `blocking`, should-fix problems go to the summary under "More problems". |
| `ignore_paths` | Glob patterns in double quotes, separated by commas, or empty | empty | Files the bot does not review. It leaves them out of the diff it reviews and drops nits and questions about them. A PR that changes only ignored files gets no automatic review. |
| `cooldown_minutes` | Whole number, 15 to 1440 | `60` | Shortest time between two reviews of one PR. A request that comes sooner waits until the time is up, and the bot replies with how long. |
| `automatic_reviews` | `on`, `off` | `on` | `on`: the bot reviews a PR once without being asked, when the PR is ready (see below). `off`: it reviews only when someone asks. |
| `title_keywords_to_skip` | Words in double quotes, separated by commas, or empty | empty | More words that stop an automatic review when they appear anywhere in a PR title, ignoring case. They add to the built-in markers below. Asking for a review still works. |
| `code_review_instructions` | File paths in double quotes, separated by commas, or empty | `AGENTS.md` | The files the bot treats as this repo's code rules and checks every change against. `AGENTS.md` means every AGENTS.md, at the root and in folders. Other paths replace it. Empty means no code rules, and the bot is told not to apply AGENTS.md, for a repo whose AGENTS.md holds instructions for other tools. Files these link to count too. |

### Format

`config.yaml` is a short list of `setting: value` lines, not full YAML. It
must still be valid YAML, because the repo's `check-yaml` pre-commit hook
checks it.

- One setting per line.
- A list goes on one line, separated by commas, in double quotes:
  `ignore_paths: "docs/**, **/*.md"`. Without the quotes, YAML rejects a
  value that starts with `*` or `[`. The bot removes the quotes.
- A YAML list (`- item` lines or `[a, b]`) does not work.
- An empty value means an empty list.
- `#` starts a comment.

The bot skips a line it cannot read, and that setting keeps its default. The
next review then ends with a note to the repo owners that names the line.

### Glob patterns for ignore_paths

Patterns start at the repo root.

- `*` matches any name inside one folder.
- `**` matches across folders.

| Pattern | Matches |
| --- | --- |
| `docs/**` | Everything under `docs/` |
| `**/*.md` | Markdown files in any folder |
| `*.md` | Markdown files at the repo root only |
| `tests/**/*.json` | JSON files anywhere under `tests/` |

### Example

```yaml
nits: inline
inline_labels: blocking
max_inline_comments: 10
ignore_paths: "docs/**, **/*.md"
cooldown_minutes: 120
title_keywords_to_skip: "experimental, on hold"
```

## When the bot reviews on its own

With `automatic_reviews: on`, the bot reviews a PR once when all of these are
true:

- The PR is not a draft.
- Its title has no built-in "not ready" marker and none of the words in
  `title_keywords_to_skip`. The built-in markers are "WIP" or "draft" at the
  start or in brackets (`[WIP]`, `(draft)`), and "do not review", "do not
  merge" or "do not submit" anywhere.
- It does not have the `skip-review-bot` label.
- A bot did not open it.
- A human reviewer is requested, and no human has reviewed it yet.

Later reviews happen only on request.

## Other controls

- **Ask for a review:** comment `@cbs-oss-review-bot` in the PR's
  Conversation tab. Repo members, collaborators and allowlisted Google
  accounts can ask.
- **Skip a PR:** add the `skip-review-bot` label. The bot skips the PR, even
  when someone asks.

## REVIEW.md

`REVIEW.md` is the guide the bot follows when it reviews a PR here: the review
passes, the rules for each folder, how to comment, and what not to flag. Each
rule names its source, such as `CONTRIBUTING.md` or a guide in
`docs/developers_guide/`. Some folders have no written rules yet; the guide
lists them. Edit it to change how the bot reviews.

- Its instructions win over the bot's own rules, except that inline comments
  still go only on lines the PR changes.
- The bot reads it from `main`, like the settings, so a PR cannot change the
  guide for its own review.
- If it defines a review format and a verdict line, the bot uses them.
  Otherwise the bot keeps its own format, which starts a clean review with
  "Looks good.".
- If the file is deleted, the bot falls back to a built-in guide kept in its
  own code.

Questions: ask yunyao-gg.
