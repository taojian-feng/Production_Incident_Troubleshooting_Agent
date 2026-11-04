---
service: orders-api
doc_type: runbook
---
# Runbook: orders-api stale tax quotes

Synthetic example, not a real production runbook.

## Symptoms
Checkout is rejected with ERR_TAX_QUOTE_STALE and logs show "tax quote expired before
capture". Orders in carts older than 15 minutes fail first.

## Diagnosis
1. Check the age of the newest row in the tax-rates cache. Older than 24 hours means the
   nightly tax-rates-sync job did not finish.
2. Check the tax-rates-sync job history for a failed run.

## Mitigation
1. Rerun the tax-rates-sync job.
2. Set the feature flag orders.tax.allow_grace_period=true to accept quotes up to one
   hour old until the sync finishes.
