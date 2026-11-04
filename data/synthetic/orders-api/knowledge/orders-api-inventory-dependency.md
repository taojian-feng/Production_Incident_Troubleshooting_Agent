---
service: orders-api
doc_type: runbook
---
# Runbook: orders-api inventory dependency failures

Synthetic example, not a real production runbook.

## Symptoms
Orders fail at the reservation step. Logs show ERR_INVENTORY_UNAVAILABLE and HTTP 503
responses from inventory-service, with latency above the 2s client timeout.

## Diagnosis
1. Check the inventory-service health dashboard and its deploy history.
2. Check whether the orders-api circuit breaker has opened. Look for
   "circuit inventory-client state=OPEN" in the logs.

## Mitigation
1. Set the feature flag orders.inventory.soft_reservation=true so orders are accepted and
   reconciled later.
2. Page the inventory on-call primary. orders-api cannot fix this failure itself.
