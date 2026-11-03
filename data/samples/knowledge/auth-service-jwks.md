---
service: auth-service
doc_type: runbook
---
# Runbook: auth-service token validation failures

## Symptoms
Spike in 401 responses across services, logs show ERR_JWKS_FETCH or
"unable to find signing key for kid".

## Diagnosis
1. Confirm whether the identity provider rotated its signing keys.
2. Check the JWKS cache TTL in auth-service config.

## Mitigation
1. Force a JWKS cache refresh via the admin endpoint /internal/jwks/refresh.
2. If the provider is down, extend the cache TTL using the emergency flag.
