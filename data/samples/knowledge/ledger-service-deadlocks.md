---
service: ledger-service
doc_type: runbook
---
# Runbook: ledger-service database deadlocks

## Symptoms
ORA-00060 deadlock detected while waiting for resource, or PostgreSQL error
"deadlock detected" (SQLSTATE 40P01) in ledger-service logs. Journal posting
latency rises and the settlement batch falls behind.

## Diagnosis
1. Run the lock-tree query in the ledger DBA notebook to find the blocking session.
2. Check whether the nightly settlement batch overlaps with the intraday
   reconciliation job; both update ledger_entries in different key orders.

## Mitigation
1. Pause the reconciliation job via the scheduler (job id recon-intraday).
2. Let the settlement batch drain, then resume reconciliation.
3. Long-term fix: enforce ordered updates by account_id in both jobs (ticket LEDG-311).
