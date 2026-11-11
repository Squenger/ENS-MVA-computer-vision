# Git notes (important)

## Why TP3 looked “already connected”

Only `TP3/recvis26_a3` was a nested git clone, with:

- `origin` → `https://github.com/fcendra-enpc/recvis26_a3` (course repo)
- local branch `main` ahead by 1 commit

That nested remote was **not** modified.

## What we did for this personal monorepo

To include TP3 **source files** in this GitHub monorepo without changing the course remote, the nested `.git` directory was **moved** (not deleted) to:

`C:\Users\aimin\Documents\ENS MVA\.git_backup_recvis26_a3`

Its `origin` remote is still exactly the course repository.

## If you need the course repo again (pull/push to the ENPC remote)

From PowerShell:

```powershell
Move-Item "C:\Users\aimin\Documents\ENS MVA\.git_backup_recvis26_a3" "C:\Users\aimin\Documents\ENS MVA\computer vision\TP3\recvis26_a3\.git"
```

Warning: restoring that nested `.git` inside the monorepo makes Git treat `TP3/recvis26_a3` as a separate repo again (parent status becomes messy). Prefer working on course submissions from a separate clone of `recvis26_a3`, and keep this monorepo for your personal backup/portfolio.
