# FinSight — n8n scheduled daily brief (Phase 5)

n8n owns **scheduled** automation: a weekday cron runs the pipeline against a
watchlist and pushes one combined brief to Slack. It calls the same
`src/api.py` `/research` endpoint the Streamlit UI does — no separate pipeline
logic. (On-demand chat access is Phase 6 / OpenClaw.)

```
Weekday 8am ─▶ Build watchlist ─▶ Call FinSight /research ─▶ Compile brief ─▶ Post to Slack
 (cron)        (env or default)     (once per query)          (fold into one    (incoming
                                                               message)          webhook)
```

## Prerequisites

1. **The FinSight API must be running and reachable.** On the host:
   ```bash
   uvicorn src.api:app --port 8000
   ```
   n8n (in Docker) reaches it at `http://host.docker.internal:8000`.
2. A **Slack incoming webhook** URL — <https://api.slack.com/messaging/webhooks>.

## Run

```bash
cd automation/n8n
cp .env.example .env          # fill in SLACK_WEBHOOK_URL (and watchlist if you want)
docker compose up -d
open http://localhost:5678     # create the local owner account on first launch
```

Then in the n8n UI:

1. **Workflows → Import from File →** `daily_brief_workflow.json`.
2. Open the workflow, click **Execute Workflow** to test it now (don't wait for
   8am). A brief should land in Slack within a few minutes — longer on a local
   Ollama model, the HTTP node waits up to 10 min per query.
3. Toggle the workflow **Active** to arm the schedule (`0 8 * * 1-5`, in
   `GENERIC_TIMEZONE`).

## Configuration

All via `automation/n8n/.env` (read by `docker-compose.yml`, surfaced to the
workflow as `$env.*`):

| Var | Purpose |
|---|---|
| `SLACK_WEBHOOK_URL` | delivery channel (required) |
| `FINSIGHT_API_URL` | where the API lives (default `http://host.docker.internal:8000`) |
| `FINSIGHT_WATCHLIST` | newline- or comma-separated queries; blank = the workflow's built-in default list |
| `GENERIC_TIMEZONE` | cron + timestamp timezone (default `Asia/Kolkata`) |

Change the schedule by editing the **Weekday 8am** node's cron expression.

## Notes

- The workflow file is hand-authored, not exported from a running instance —
  n8n may upgrade stale node `typeVersion`s on import; let it.
- A `/research` call that fails (`502`, timeout) doesn't abort the brief — that
  query's section shows the error instead (`onError: continueRegularOutput`).
- The brief posts each report's **Executive Summary** (or the first ~700 chars)
  to keep the Slack message readable; the full report stays in the API response.
- Bundling n8n + API + Streamlit into one root `docker-compose.yml` is the
  Phase 7 stretch goal; this compose file is n8n only.
