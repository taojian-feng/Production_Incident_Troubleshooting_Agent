---
service: payments-api
doc_type: runbook
---
# Runbook: payments-api upstream gateway timeouts

## Symptoms
Alarms on payments-api p99 latency above 2s and HTTP 504 rate above 1%. Logs show
ERR_GATEWAY_TIMEOUT from the card-network adapter and connection pool exhaustion
messages such as "HikariPool-1 - Connection is not available, request timed out".

## Diagnosis
1. Check the CardNetworkLatency dashboard; if the upstream acquirer latency is
   elevated, the problem is external. Open a ticket with the acquirer NOC.
2. Check the RDS proxy connection count. A value at max_connections means the
   pool is exhausted, usually after a deploy that raised worker concurrency.
3. Compare the current task definition revision to the previous one.

## Mitigation
1. If a deploy happened in the last hour, roll back the payments-api ECS service
   to the previous task definition revision.
2. If the acquirer is degraded, enable the circuit breaker flag
   payments.acquirer.circuit_breaker=true in AppConfig to fail fast and queue retries.
3. Scale RDS proxy max_connections_percent to 90 only as a temporary measure.

## Escalation
Page the payments on-call secondary if the 504 rate stays above 5% for 10 minutes.
