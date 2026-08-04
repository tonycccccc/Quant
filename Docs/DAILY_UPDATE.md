# Daily Update System

Automated daily maintenance for the trading pipeline. Keeps `raw_bars.parquet`, `features.parquet`, `macro_features.parquet`, and `quant_model.pkl` fresh so live trading always sees current-day-accurate data and never drifts stale.

## The one-liner

```bash
python Code/main.py daily-update
```

## What it does

Runs a 4-step idempotent pipeline in ~5-10 min:

| Step | Action | Time | Frequency |
|---|---|---|---|
| 1. Fetch incremental | Only new bars since last cache | ~30 sec | Every day |
| 2. Rebuild features | Full features.parquet rebuild | ~5 min | Every day |
| 3. Check model age | Compare against `--retrain-days` (default 7) | <1 sec | Every day |
| 4. Retrain model | Only if model ≥ N days old | ~5 min | Every N days |

Safe to run multiple times per day — the fetch is a no-op if bars are already current.

## CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--lookback DAYS` | `2` | Days of overlap on incremental fetch (catches edge-case missed bars) |
| `--retrain-days N` | `7` | Trigger retrain when model this old |
| `--force-retrain` | off | Retrain regardless of age |

## Scheduling

### Windows Task Scheduler

Open PowerShell as admin, then:

```powershell
$action = New-ScheduledTaskAction `
  -Execute "C:\Users\chenz\miniconda3\python.exe" `
  -Argument "C:\Users\chenz\Desktop\AI_Trading_Project\Code\main.py daily-update" `
  -WorkingDirectory "C:\Users\chenz\Desktop\AI_Trading_Project"

$trigger = New-ScheduledTaskTrigger -Daily -At 08:00
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable  # catches missed runs (weekends, PC asleep)

Register-ScheduledTask `
  -TaskName "AI-Trader-Daily-Update" `
  -Description "Fetches new bars, rebuilds features, weekly-retrains ML model" `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -RunLevel Highest
```

Verify:
```powershell
Get-ScheduledTask -TaskName "AI-Trader-Daily-Update"
Start-ScheduledTask -TaskName "AI-Trader-Daily-Update"   # run now
```

Logs land in `Logs/daily_update.log` (whichever redirection you configure). Add to the scheduled task with `-Argument "... > Logs\daily_update.log 2>&1"` inside a wrapper `.bat` if you want file logging.

### Linux/macOS cron

```cron
# Daily at 8:00 AM ET (adjust for TZ)
0 8 * * 1-5 cd /path/to/AI_Trading_Project && /path/to/python Code/main.py daily-update >> Logs/daily_update.log 2>&1
```

Trading days only (Mon-Fri). Adjust the `TZ` in your crontab if the host is on UTC.

### AWS EventBridge + Lambda (production deploy)

Recommended long-term deployment target for 24/7 uptime.

**Option A — EC2 + cron:**
```bash
# On the EC2 instance
crontab -e
# Add:
0 12 * * 1-5 cd /home/ec2-user/ai-trading && /home/ec2-user/miniconda3/bin/python Code/main.py daily-update >> Logs/daily_update.log 2>&1
# (12 UTC = 8 AM ET; adjust for DST)
```

**Option B — Lambda + EventBridge (serverless):**
```
EventBridge rule:
  Schedule expression: cron(0 12 ? * MON-FRI *)   # 8 AM ET, weekdays
  Target: Lambda function `AI-Trader-Daily-Update`

Lambda:
  Runtime: Python 3.11 (custom Docker image with LightGBM + pandas)
  Timeout: 15 min (max for Lambda; use ECS Task if you need more)
  Memory: 3 GB (feature build needs it)
  Environment vars: ALPACA_API_KEY, ALPACA_SECRET_KEY, Discord_Webhook, OPEN_ROUTER_API_KEY
  Storage: EFS mounted at /mnt/models for Models/ persistence across invocations
  Handler: from ml.main_lambda import daily_update_handler
```

Then create a thin Lambda handler:
```python
# Code/lambda_handlers.py
def daily_update_handler(event, context):
    import subprocess
    result = subprocess.run(
        ['python', '/var/task/Code/main.py', 'daily-update'],
        capture_output=True, text=True, timeout=850,
    )
    return {
        'statusCode': 200 if result.returncode == 0 else 500,
        'body': result.stdout + result.stderr,
    }
```

## Verification

After the first scheduled run:

```bash
# Check the log
cat Logs/daily_update.log | tail -30

# Verify files are fresh
python -c "
import pandas as pd
from datetime import datetime
raw = pd.read_parquet('Models/raw_bars.parquet')
feat = pd.read_parquet('Models/features.parquet')
print(f'raw_bars last: {raw.index.get_level_values(1).max()}')
print(f'features last: {feat.index.max()}')
print(f'now:           {datetime.now()}')
"
```

## Failure recovery

If a run fails partway (e.g., during the feature rebuild):

1. `raw_bars.parquet` is safe — the fetch is atomic (writes to disk only after complete concat).
2. `features.parquet` may be stale if step 2 failed. Just re-run `daily-update` — step 2 always does a full rebuild.
3. `quant_model.pkl` is only overwritten on successful save at end of training — never partial.

Rollback: `git checkout Models/quant_model.pkl` if you tracked it, or fall back to the alert-log-based comparison to detect and roll back a bad retrain.

## Alerts on failure

Wire the scheduled task to Discord notification on error:

```powershell
# Inside a wrapper script daily_update.ps1
try {
    python Code/main.py daily-update
    if ($LASTEXITCODE -ne 0) { throw "Exit code $LASTEXITCODE" }
} catch {
    $body = @{ content = "❌ Daily update FAILED: $_" } | ConvertTo-Json
    Invoke-RestMethod -Uri "$env:Discord_Webhook" -Method Post -ContentType "application/json" -Body $body
}
```

## When to change `--retrain-days`

- **7 days (default)**: monthly-market use; ML model stays reasonably fresh.
- **3 days**: aggressive; if regime changes fast (correction period), model adapts faster.
- **30 days**: conservative; if training is slow or you're afraid of overfitting on recent noise.

Weekly retrain aligns with monthly regime tracking. Don't retrain daily — the OOS walk-forward showed the model is stable for 6+ months at a time.
