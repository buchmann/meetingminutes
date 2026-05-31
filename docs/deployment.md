# Build & Deployment

How Transkriptor is built into a container image and deployed to the Kubernetes
cluster. This is the "remote-backend" deployment: all heavy ML work (WhisperX
transcription, vLLM summarization) runs on the DGX Spark, and the container only
runs the FastAPI web app.

---

## Topology

```
  Mac (source of truth)                linux  (build server)            k8s cluster (master 192.168.178.35)
  /Users/manfred/claude/Code/   rsync   /home/manfred/transkriptor-build  ns: transkriptor
      transkriptor-app          ─────►   docker build (amd64)             Deployment → Pod (worker-node1)
                                          docker push  ──► Docker Hub ──►  image pull: mbx1010/transkriptor:latest
                                                          (mbx1010)        Ingress: transkriptor.lab.allwaysbeginner.com
                                                                            └─► DGX Spark 192.168.178.190 (whisperx/vLLM/Instana)
```

| Component | Value |
|-----------|-------|
| Build server | `linux` (`linux.lab.allwaysbeginner.com`), user `manfred`, Docker 29.2.1 |
| Build checkout | `/home/manfred/transkriptor-build` |
| Image | `docker.io/mbx1010/transkriptor` — tags `latest`, `multiuser` |
| Cluster API | `https://192.168.178.35:6443` (nodes: `master-node`, `worker-node1`, `worker-node2`) |
| Namespace | `transkriptor` |
| Storage | PVC `transkriptor-data` (20Gi, `local-path`) mounted at `/app/data` |
| Ingress | nginx, host `transkriptor.lab.allwaysbeginner.com` (plain HTTP) |
| Backends (DGX) | whisperx `:8003`, vLLM `:8001`, gpu-manager `:9090`, Instana OTLP `:4328` |

> **Why build on `linux`?** The cluster nodes are `amd64`; the Mac is `arm64`.
> Building natively on the `linux` host produces an `amd64` image that the
> cluster can run without emulation.

---

## Prerequisites

**On the `linux` build server (one-time):**
- Docker installed and running.
- Logged in to Docker Hub as the image owner: `docker login` (account `mbx1010`).
- A checkout/working copy at `/home/manfred/transkriptor-build`.

**For deploying:**
- `kubectl` with a valid kubeconfig for `192.168.178.35` (context `kubernetes-admin@kubernetes`), **or** another admin path to the cluster.

---

## 1. Build the image

The Dockerfile installs only the slim **remote** dependency set
(`requirements-remote.txt` — no torch/whisper/pyannote), installs the app, and
runs `python -m transkriptor`. No code changes are needed between builds beyond
syncing the source.

```bash
# From the Mac: push the current working tree to the build server
# (excludes runtime data, secrets, git, backups)
rsync -az \
  --exclude 'data/' --exclude 'data.backup-*/' --exclude 'backups/' \
  --exclude '.git/' --exclude '.env' --exclude '__pycache__/' \
  --exclude '*.egg-info/' --exclude '*.m4a' --exclude '.venv/' \
  /Users/manfred/claude/Code/transkriptor-app/  linux:/home/manfred/transkriptor-build/

# On the build server: build (amd64) and push
ssh linux '
  cd ~/transkriptor-build &&
  docker build -t mbx1010/transkriptor:latest -t mbx1010/transkriptor:multiuser . &&
  docker push mbx1010/transkriptor:latest &&
  docker push mbx1010/transkriptor:multiuser
'
```

The deployment uses `imagePullPolicy: Always` on the `:latest` tag, so a rollout
restart always pulls the freshly pushed image. Use an extra dated/semantic tag
(e.g. `:multiuser`) for traceability and easy rollback.

---

## 2. Kubernetes resources

All manifests live in [`k8s/`](../k8s). They are environment config only — no
application code.

| File | Resource | Purpose |
|------|----------|---------|
| `namespace.yaml` | Namespace `transkriptor` | Isolation |
| `pvc.yaml` | PVC `transkriptor-data` (20Gi `local-path`) | Persists SQLite DB + uploads/outputs at `/app/data` |
| `configmap.yaml` | ConfigMap `transkriptor-config` | All non-secret env (backends, OTel, session settings) |
| `secret.example.yaml` | Secret template | **Real secret applied out-of-band** (admin password) |
| `deployment.yaml` | Deployment `transkriptor` | 1 replica, mounts ConfigMap + Secret via `envFrom`, PVC at `/app/data`, liveness `/api/livez`, readiness `/api/readyz` |
| `service.yaml` | Service (ClusterIP) | `:80 → :8000` |
| `ingress.yaml` | Ingress (nginx) | `transkriptor.lab.allwaysbeginner.com → service:80`, 2 GB body limit, long timeouts for large uploads |

### Multi-user configuration

Authentication is configured via env (see the [Multi-user section of the README](../README.md#multi-user--authentication)):

- **Secret `transkriptor-secrets`** holds the admin password (never committed):
  ```bash
  kubectl -n transkriptor create secret generic transkriptor-secrets \
    --from-literal=TRANSKRIPTOR_ADMIN_PASSWORD='<strong-password>' \
    --dry-run=client -o yaml | kubectl apply -f -
  ```
- **ConfigMap** adds (non-secret) auth settings:
  ```yaml
  TRANSKRIPTOR_ADMIN_USERNAME: "admin"
  TRANSKRIPTOR_SESSION_TTL_HOURS: "720"
  TRANSKRIPTOR_SESSION_COOKIE_SECURE: "false"   # ingress is plain HTTP
  ```
- **Deployment** consumes both:
  ```yaml
  envFrom:
    - configMapRef: { name: transkriptor-config }
    - secretRef:    { name: transkriptor-secrets }
  ```

> Set `TRANSKRIPTOR_SESSION_COOKIE_SECURE: "true"` if/when the ingress is moved
> behind TLS, otherwise the session cookie won't be sent and login will fail.

---

## 3. Deploy / rollout

### First-time install
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
# Secret: create with a real password (see Multi-user section above) — do NOT apply secret.example.yaml as-is
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/deployment.yaml
```

### Update to a new image build
```bash
# After build+push (step 1):
kubectl -n transkriptor rollout restart deployment/transkriptor
kubectl -n transkriptor rollout status  deployment/transkriptor
```
(If you only changed manifests, `kubectl apply -f k8s/<file>.yaml`; changing the
ConfigMap/Secret does **not** auto-restart pods — follow with a rollout restart.)

### First start of the multi-user image — automatic migration
On the first start of the multi-user image against an existing `/app/data` DB,
the app:
1. Adds `user_id` / `visibility` columns + indexes to the `jobs` table.
2. **Deletes legacy (owner-less) jobs** and prunes their upload/output dirs
   (clean multi-user start — back up the PVC first if you need the old data).
3. **Seeds the admin** account from `TRANSKRIPTOR_ADMIN_USERNAME` /
   `TRANSKRIPTOR_ADMIN_PASSWORD` (only when no users exist yet).

Confirm in the logs:
```
Pruned orphan directory /app/data/uploads/...
Seeded initial admin user 'admin'
Application startup complete.
```

---

## 4. Verify

```bash
kubectl -n transkriptor get pods,svc,ingress
kubectl -n transkriptor logs deploy/transkriptor --tail=30

B=http://transkriptor.lab.allwaysbeginner.com
curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' $B/        # 303 -> /login
curl -s -o /dev/null -w '%{http_code}\n' $B/login                  # 200
curl -s -o /dev/null -w '%{http_code}\n' $B/api/jobs               # 401 (unauth)
# Login + authenticated check:
curl -s -c /tmp/c -d "username=admin&password=<admin-pw>" $B/login
curl -s -b /tmp/c $B/api/jobs                                      # [] (200)
```
Liveness/readiness probes are at `/api/livez` and `/api/readyz` (no auth).

---

## 5. Common operations

| Task | Command |
|------|---------|
| Tail logs | `kubectl -n transkriptor logs deploy/transkriptor -f` |
| Restart (re-pull `:latest`) | `kubectl -n transkriptor rollout restart deploy/transkriptor` |
| Roll back | `kubectl -n transkriptor rollout undo deploy/transkriptor` (or set image to a known tag, e.g. `:multiuser`) |
| Change admin password | re-create the Secret (above) **and** reset in-app at `/admin/users`; restart to re-read the Secret if relying on the seed |
| Add/remove users | in-app at `/admin/users` (admin only) — no manifest change |
| Inspect the DB | `kubectl -n transkriptor exec deploy/transkriptor -- python -c "import sqlite3;print(sqlite3.connect('/app/data/transkriptor.db').execute('select count(*) from jobs').fetchone())"` |
| Scale | keep `replicas: 1` — the SQLite DB on a `ReadWriteOnce` PVC is single-writer; do not scale out without switching to a shared DB |

---

## Notes & gotchas

- **Single replica only.** State is SQLite on a `ReadWriteOnce` PVC. Running >1
  replica would corrupt data and split sessions. For HA, move to Postgres + a
  shared session store first.
- **Secrets are not committed.** `k8s/secret.example.yaml` is a template; the
  real `transkriptor-secrets` is applied out-of-band.
- **Build server kubectl.** At time of writing, `linux`'s kubeconfig token for
  the cluster is expired — the last rollout was applied from an admin kubeconfig
  on `192.168.178.35`. Refresh `linux`'s kubeconfig if you want to build *and*
  deploy from the build server in one place.
- **Backends are external.** The container is stateless compute-wise; it calls
  the DGX Spark (`192.168.178.190`) for transcription, summarization, and ships
  traces to the Instana agent. If those are down, jobs fail but the web app and
  auth still work.
