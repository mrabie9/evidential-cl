# Shell Command Habits

These rules exist to avoid manual verification prompts during autonomous execution.
Violating them will cause Claude Code to pause and require human approval.

---

## Hard rules

- **NEVER** use `source` in any form — not `source venv/bin/activate`, not `source .env`,
  not `source anything`. It always triggers a verification prompt.
- **NEVER** chain `cd /path; command` or `cd /path && command`. Use absolute paths or
  `git -C` instead.
- **NEVER** write complex inline shell logic (variable assignment + conditional + command
  substitution in a single string). The static analyser cannot follow it and will block
  execution every time.
- **NEVER** embed inline bash comments (`# ...`) inside quoted shell strings passed to
  `bash -c "..."`. Put comments in prose above the command instead.

---

## Repo paths (La-MAML)

| Resource | Path |
|---|---|
| Repo root | `/home/lunet/wsmr11/repos/La-MAML` |
| Venv root | `/home/lunet/wsmr11/repos/La-MAML/la-maml_env` |
| Python | `/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python` |
| pip | `/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/pip` |
| ruff | `/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/ruff` |
| black | `/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/black` |
| pytest | `/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/pytest` |

---

## Substitution patterns

### Running Python / experiments

```bash
# WRONG
cd /home/lunet/wsmr11/repos/La-MAML; source la-maml_env/bin/activate; python main.py ...

# CORRECT
timeout 400 /home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/python \
  /home/lunet/wsmr11/repos/La-MAML/main.py \
  --model lamaml_cifar ...
```

### Linting and formatting

```bash
# WRONG
cd /home/lunet/wsmr11/repos/La-MAML; source la-maml_env/bin/activate; ruff check model/x.py && black --check model/x.py

# CORRECT
/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/ruff check \
  /home/lunet/wsmr11/repos/La-MAML/model/x.py && \
/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/black --check \
  /home/lunet/wsmr11/repos/La-MAML/model/x.py 2>&1 | tail -1
```

### Git operations

```bash
# WRONG
cd /home/lunet/wsmr11/repos/La-MAML; git add model/x.py; git status --short model/x.py

# CORRECT
git -C /home/lunet/wsmr11/repos/La-MAML add model/x.py
git -C /home/lunet/wsmr11/repos/La-MAML status --short model/x.py
```

### Complex shell logic (find, conditionals, loops)

```bash
# WRONG — cannot be statically analysed
cd /repo; F=$(find logs/ -name terminal.log -newermt "-10 minutes" | head -1); [ -n "$F" ] && grep "Task" "$F"

# CORRECT — write to a script file, then execute it
cat > /tmp/check_log.sh << 'EOF'
#!/bin/bash
F=$(find /home/lunet/wsmr11/repos/La-MAML/logs -name terminal.log -newermt "-10 minutes" 2>/dev/null | head -1)
echo "F=$F"
[ -n "$F" ] && grep -iE "Task 0:|Ep 1/1|Epoch Time" "$F" | tail -5
EOF
bash /tmp/check_log.sh
```

### Installing packages

```bash
# WRONG
source la-maml_env/bin/activate && pip install x

# CORRECT
/home/lunet/wsmr11/repos/La-MAML/la-maml_env/bin/pip install x
```

---

## Quick reference

| Situation | Safe pattern |
|---|---|
| Activate venv | Don't. Use the binary directly. |
| Run Python | `/path/to/venv/bin/python /path/to/script.py` |
| Run tool (ruff, black, pytest) | `/path/to/venv/bin/toolname` |
| Change directory | Use absolute paths or `git -C /path` |
| Complex shell logic | Write to `/tmp/script.sh`, run `bash /tmp/script.sh` |
| Inline comment in shell string | Remove it, or move it to prose above the command |