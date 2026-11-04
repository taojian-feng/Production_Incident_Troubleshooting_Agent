---
service: orders-api
doc_type: runbook
---
# Runbook: orders-api cache timeouts and 5xx errors

Synthetic example, not a real production runbook.

## Symptoms
Alarm on orders-api 5xx rate above 2% for one minute. Logs show ERR_CACHE_TIMEOUT from
the session cache client and "RedisCommandTimeoutException: command timed out after
500ms". Checkout requests return HTTP 503 because the request path has no cache fallback.

## Diagnosis
1. Check the ElastiCache events for orders-cache. A recent failover or node replacement
   means clients may still hold connections to the old primary.
2. Check the orders-api logs for "READONLY You can't write against a read only replica".
   This confirms that clients are writing to a demoted primary.
3. Check CurrConnections and EngineCPUUtilization on orders-cache. CPU above 90% with no
   failover points to a hot key or a traffic spike instead.
4. If the logs show ERR_INVENTORY_UNAVAILABLE instead, follow the inventory dependency
   runbook. The cause is downstream and not the cache.

## Mitigation
1. After a failover, restart the orders-api ECS service with a rolling deployment so
   clients reconnect through the primary endpoint. Do not point clients at a node endpoint.
2. For a hot key or traffic spike, scale orders-cache to the next node size and set the
   feature flag orders.cache.read_through=false to shed cache reads.
3. Do not flush orders-cache during an incident. Flushing drops active checkout sessions.

## Escalation
Page the orders on-call secondary if the 5xx rate stays above 5% for 10 minutes.
