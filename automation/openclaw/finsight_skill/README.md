# FinSight OpenClaw AgentSkill — design notes (unverified placeholder)

**Read this before writing any OpenClaw skill code.** Everything below is a
design intent, not a confirmed schema. I (Claude, drafting this PRD) do not
have hands-on access to OpenClaw's current AgentSkill manifest format —
don't assume the shape described here matches what OpenClaw actually
expects. Before implementing:

1. Fetch OpenClaw's current documentation/repo (github.com/openclaw/openclaw)
   and find the AgentSkill authoring guide.
2. Confirm: the manifest file format and required fields, how a skill
   declares its trigger phrase(s), how a skill makes an outbound HTTP call,
   and how the response gets routed back to the originating chat (WhatsApp/
   Telegram/etc.).
3. Only then write the actual skill files — replace this README's "intended
   design" section with the real, verified structure.

## Intended design (pending verification above)

- **Trigger:** a phrase like "finsight <question>" or a dedicated command,
  invoked from any connected messaging channel (WhatsApp/Telegram/Slack —
  whichever OpenClaw is configured with).
- **Action:** the skill takes the user's question as free text, does a
  simple HTTP POST to the FinSight FastAPI endpoint (`src/api.py`,
  `/research`), and returns the `report` field of the response back into
  the chat.
- **No local LLM reasoning inside the skill itself** — OpenClaw's own agent
  layer handles the conversational wrapper (understanding "finsight, how's
  KPIT looking?" as an invocation with `query="how's KPIT looking?""`), but
  the actual research work happens entirely in FinSight's own FastAPI
  service. The skill is a thin HTTP bridge, not a second implementation of
  the pipeline — consistent with the "one backend, three interfaces"
  principle in PRD.md Section 1.
- **Long-running consideration:** the pipeline can take a while (multiple
  agent calls + tool calls). Check whether OpenClaw's skill execution model
  supports async/long-running actions with a "still working..." interim
  reply, or whether the HTTP call needs a timeout-and-poll pattern instead
  of a blocking request — this depends on OpenClaw's actual execution model
  (verify, don't assume).
- **Config:** the FastAPI base URL should be configurable (env var or
  OpenClaw's own skill config mechanism, whichever is idiomatic once you've
  read the real docs), not hardcoded to `localhost` — OpenClaw and the
  FinSight API may run on different hosts/containers depending on how each
  is self-hosted.

## Why this exists (for the resume narrative, see PRD.md Section 1)

n8n handles *scheduled* delivery. This skill handles *on-demand*
conversational access — the same backend, a different trigger. Keep that
distinction explicit in the README once this is built.
