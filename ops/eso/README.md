# External Secrets Operator (ESO) → Vault wildcard distribution

Automates distributing the Let's Encrypt **wildcard** (`*.lab.allwaysbeginner.com`,
issued centrally by acme.sh on kubeflow, stored in Vault KV
`secret/certs/wildcard-allwaysbeginner`) into per-namespace `kubernetes.io/tls`
secrets on the main cluster (192.168.178.35). Replaces the manual
`kubectl create secret tls` copy used for local-ai / bum / gitlab.

## Why ESO and not cert-manager
cert-manager **issues** certs (ACME/LE, or from a CA via its PKI/Vault-PKI issuer).
It cannot pull an already-issued cert out of Vault **KV**, and a Vault **PKI** issuer
would only mint a *private* CA cert (no public browser trust). The wildcard is public
LE, issued once centrally (to dodge LE's 5/week duplicate-cert rate limit) and just
needs **distributing** — that's ESO's job. The two are complementary; they don't
conflict. cert-manager stays available (letsencrypt-prod/staging, selfsigned) for any
service that genuinely needs to *issue* a cert.

## State
- ESO installed via Helm: `helm install external-secrets external-secrets/external-secrets -n external-secrets --create-namespace --set installCRDs=true`.
- Manifests here: `clustersecretstore-vault.yaml` (Vault AppRole + CA baked in) and
  `externalsecret-<ns>.yaml` for local-ai / bum / gitlab.

## One-time Vault setup (run on kubeflow / 192.168.178.97, authenticated)
```bash
vault policy write eso-certs - <<'EOF'
path "secret/data/certs/*"     { capabilities = ["read"] }
path "secret/metadata/certs/*" { capabilities = ["read","list"] }
EOF
vault auth enable approle 2>/dev/null || true
vault write auth/approle/role/eso token_policies="eso-certs" \
     token_ttl=20m token_max_ttl=1h secret_id_num_uses=0 secret_id_ttl=0
vault read   auth/approle/role/eso/role-id        # -> role_id
vault write -f auth/approle/role/eso/secret-id    # -> secret_id
```

## Wire it up (on the main cluster, 192.168.178.35)
```bash
# 1) secret_id into a k8s secret
kubectl create secret generic vault-approle -n external-secrets \
  --from-literal=secret_id='<SECRET_ID>'
# 2) put role_id into clustersecretstore-vault.yaml (REPLACE_WITH_ROLE_ID), then:
kubectl apply -f ops/eso/clustersecretstore-vault.yaml
kubectl apply -f ops/eso/externalsecret-local-ai.yaml
kubectl apply -f ops/eso/externalsecret-bum.yaml      # optional: adopt other svcs
kubectl apply -f ops/eso/externalsecret-gitlab.yaml
# 3) verify
kubectl get clustersecretstore vault-kv
kubectl get externalsecret -A          # want STATUS=SecretSynced / READY=True
```

Once `SecretSynced`, the TLS secrets refresh from Vault automatically (refreshInterval
1h) — no more manual copy on renewal. role_id is not secret; the secret_id is (rotate
with `vault write -f auth/approle/role/eso/secret-id` and update the k8s secret).
The Vault CA (self-signed `O=HashiCorp, CN=Vault`) is baked into the ClusterSecretStore
`caBundle`; if Vault's cert changes, refresh it.
