---
service: payments-api
doc_type: postmortem
---
# Post-mortem: payments-api connection pool exhaustion (July 2026)

## Summary
On 2026-07-14 payments-api returned HTTP 504 to 7% of card authorisations for 38
minutes. Incident INC0041877.

## Root cause
Deploy 2026.07.14-2 raised Gunicorn workers from 8 to 16 per task without raising
the RDS proxy connection limit, exhausting the pool under peak load.

## What went well
The rollback runbook worked; mitigation took 6 minutes once the deploy was identified.

## Action items
Add a pre-deploy check that worker count times tasks stays below the proxy limit.
