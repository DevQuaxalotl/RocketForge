#!/usr/bin/env bash
# RocketForge training entrypoint.
#
# Runs ForgeConsoleTrainer, forwarding graceful-stop signals so the trainer's
# own save-and-exit runs, then exports the files its stage queue touches (see sync-manifest) to Cloudflare R2
# (skipped when R2 is not configured). The queue segments' periodic autosaves (temp_checkpoints/)
# ride along on every export as a crash backup: each launch pulls the newest one per segment,
# which the trainer promotes to a durable checkpoint, then wipes the remote copies before training starts. See:
#   docs/superpowers/specs/2026-05-31-remote-training-checkpoint-export-design.md
#
# NOTE: `set -e` is intentionally NOT used — a failed export must not prevent us
# from propagating the trainer's real exit code.
set -uo pipefail

SRC_ROOT="/workspace/RocketForge"
# Container-local capture target for the run log (see the exec-tee in main()).
# Uploaded to R2 as a single fixed-name object, overwritten every run.
LOG_FILE="/workspace/training.log"

# Cloudflare R2 API tokens are scoped to a bucket with *Object* Read & Write and
# CANNOT create buckets. rclone's default pre-upload step issues a CreateBucket call
# to "ensure the bucket exists", which R2 rejects with 403 AccessDenied and aborts
# the upload — even though the bucket already exists. Disabling that check makes every
# rclone upload in this container (both run_export and run_export_log) work with a
# scoped token. Harmless: the bucket is created once in the dashboard during setup, so
# there is never anything to auto-create. Matches the existing RCLONE_CONFIG_R2_* family
# so it binds to the `r2` remote.
export RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true

# Common flags for every manifest-driven copy. --update never clobbers a newer file.
# --fast-list is deliberately absent: it lists the whole tree before filtering,
# which is exactly the walk the manifest exists to avoid.
RCLONE_SYNC_FLAGS=(--update --transfers 8 --checkers 16 -P)

# The learner writes a temp checkpoint as several files over a few seconds;
# skipping anything this fresh keeps a half-written checkpoint off R2 (it goes up next cycle).
TEMP_CHECKPOINT_MIN_AGE=2m

# Prints the value of the start command's `--agent X`, or nothing when it names no agent
# (the trainer then uses the GUI-selected agent), with or without a leading `train` command word.
# The CLI has no --opt=value form, so only the separate-word form is recognised.
resolve_agent_arg() {
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--agent" && $# -ge 2 ]]; then
            echo "$2"
            return 0
        fi
        shift
    done
}

# Writes the trainer's rclone filter list for the selected build to a temp file and prints its path.
# $1 = optional agent (empty = GUI-selected), $2 = optional extra sync-manifest flag (e.g. --metadata-only).
# Returns non-zero (and prints nothing) if the manifest could not be computed.
write_manifest() {
    local agent="${1:-}" mode="${2:-}" file
    file="$(mktemp)" || return 1
    if ! ForgeConsoleTrainer sync-manifest ${agent:+--agent "$agent"} ${mode} > "$file"; then
        rm -f "$file"
        return 1
    fi
    if [[ ! -s "$file" ]]; then rm -f "$file"; return 1; fi
    echo "$file"
}

# Export the files the selected build's stage queue touches to R2 (see sync-manifest):
# the segments it continues or created, the lineage's pools, and metadata.
# Uses `copy` (never deletes) + `--update` (never clobbers a newer remote file).
# Returns non-zero if the copy failed, so the caller can gate pod self-termination on a successful export.
# Recomputed on every call, so a queue that advanced to its next stage is picked up.
# $1 = optional agent.
run_export() {
    if [[ -z "${R2_BUCKET:-}" ]]; then
        echo "[entrypoint] R2_BUCKET not set — skipping checkpoint export."
        return 0
    fi
    local filter
    if ! filter="$(write_manifest "${1:-}")"; then
        echo "[entrypoint] WARNING: could not compute the sync manifest — skipping export."
        return 1
    fi
    local rc=0
    echo "[entrypoint] Exporting training manifest to r2:${R2_BUCKET}/Bots ..."
    if ! rclone copy "${SRC_ROOT}/Bots" "r2:${R2_BUCKET}/Bots" --filter-from "$filter" "${RCLONE_SYNC_FLAGS[@]}"; then
        echo "[entrypoint]   WARNING: export failed."
        rc=1
    fi
    rm -f "$filter"
    run_export_temp_checkpoints "${1:-}"
    echo "[entrypoint] Export complete."
    return $rc
}

# Push the queue segments' temp checkpoints (the tsPerSave autosaves) to R2 so a lost
# instance leaves its latest autosave recoverable by hand. Non-critical: a failure warns
# and is ignored — it never affects export_ok or the self-terminate gate.
# $1 = optional agent.
run_export_temp_checkpoints() {
    local filter
    if ! filter="$(write_manifest "${1:-}" --temp-checkpoints)"; then
        echo "[entrypoint] WARNING: could not compute the temp-checkpoint manifest — skipping."
        return 0
    fi
    echo "[entrypoint] Exporting temp checkpoints to r2:${R2_BUCKET}/Bots ..."
    rclone copy "${SRC_ROOT}/Bots" "r2:${R2_BUCKET}/Bots" --filter-from "$filter" \
        --min-age "$TEMP_CHECKPOINT_MIN_AGE" "${RCLONE_SYNC_FLAGS[@]}" \
        || echo "[entrypoint]   WARNING: temp checkpoint export failed."
    rm -f "$filter"
}

# Pull each queue segment's newest remote autosave before run_prune_temp_checkpoints deletes them,
# so a run that ended without a durable save (a crash, preemption, or SIGKILL) resumes from it.
# The trainer promotes a resumed autosave to a durable checkpoint, which the regular export then carries.
# Non-fatal: on failure the run resumes from the newest durable checkpoint.
# $1 = optional agent.
run_seed_newest_temp_checkpoints() {
    if [[ -z "${R2_BUCKET:-}" ]]; then
        return 0
    fi
    local filter listing
    if ! filter="$(write_manifest "${1:-}" --temp-checkpoints)"; then
        echo "[entrypoint] WARNING: could not compute the temp-checkpoint manifest — skipping autosave seed."
        return 0
    fi
    if ! listing="$(rclone lsf -R --files-only --filter-from "$filter" "r2:${R2_BUCKET}/Bots")"; then
        echo "[entrypoint] WARNING: could not list remote temp checkpoints — skipping autosave seed."
        rm -f "$filter"
        return 0
    fi
    rm -f "$filter"

    # Listed paths look like <segment>/temp_checkpoints/<timesteps>/POLICY.lt; keep the highest timesteps per segment.
    local newest checkpoint
    newest="$(sed -nE 's#^(.+/temp_checkpoints)/([0-9]+)/POLICY\.lt$#\1\t\2#p' <<< "$listing" \
        | sort -t $'\t' -k1,1 -k2,2n \
        | awk -F '\t' '{ newest[$1] = $2 } END { for (root in newest) print root "/" newest[root] }')"
    while IFS= read -r checkpoint; do
        [[ -n "$checkpoint" ]] || continue
        echo "[entrypoint] Seeding newest autosave ${checkpoint} from r2:${R2_BUCKET}/Bots ..."
        rclone copy "r2:${R2_BUCKET}/Bots/${checkpoint}" "${SRC_ROOT}/Bots/${checkpoint}" "${RCLONE_SYNC_FLAGS[@]}" \
            || echo "[entrypoint] WARNING: autosave seed failed — resuming from the newest durable checkpoint."
    done <<< "$newest"
}

# Delete the queue segments' temp checkpoints from R2 before training starts.
# Exports never delete (copy --update), so this is the only thing that stops the
# autosaves from accumulating one checkpoint per tsPerSave forever.
# Runs after run_seed_newest_temp_checkpoints, so the newest autosave survives locally and is promoted on resume.
# Non-fatal: a failure only means stale autosaves linger until the next launch.
# $1 = optional agent.
run_prune_temp_checkpoints() {
    if [[ -z "${R2_BUCKET:-}" ]]; then
        return 0
    fi
    local filter
    if ! filter="$(write_manifest "${1:-}" --temp-checkpoints)"; then
        echo "[entrypoint] WARNING: could not compute the temp-checkpoint manifest — skipping prune."
        return 0
    fi
    echo "[entrypoint] Pruning the previous run's temp checkpoints from r2:${R2_BUCKET}/Bots ..."
    rclone delete "r2:${R2_BUCKET}/Bots" --filter-from "$filter" \
        || echo "[entrypoint] WARNING: temp checkpoint prune failed."
    rm -f "$filter"
}

# Upload the most-recent run log to R2 as a single fixed-name object, overwriting
# any previous one. Non-critical: a failure warns and is ignored — it must never
# affect export_ok or the self-terminate gate (only checkpoint export does that).
# Uses `copyto` (not `copy`) because the local basename (training.log) differs from
# the remote name (latest-training.log); no `--update`, so it always overwrites.
run_export_log() {
    # Silent skip (no echo): on the final path run_export already prints the
    # R2-unset notice just above this, and a second one would be redundant noise.
    if [[ -z "${R2_BUCKET:-}" ]]; then
        return 0
    fi
    [[ -f "$LOG_FILE" ]] || return 0
    echo "[entrypoint] Uploading training log to r2:${R2_BUCKET}/latest-training.log ..."
    rclone copyto "$LOG_FILE" "r2:${R2_BUCKET}/latest-training.log" \
        || echo "[entrypoint] WARNING: training log upload failed."
}

# Pull what the selected agent's stage queue needs from R2 before training
# so a cloud run CONTINUES an in-progress stage instead of starting fresh.
# Three passes: forge_settings.json (it names the segment a previous run created, so a duplicate is never created),
# the build's metadata (segment.json files the lineage walk reads), then the full manifest.
# The pod disk starts empty (Models excluded from the image), so a plain `copy` is correct and non-destructive.
# Non-fatal: on any failure we simply start from scratch.
# Args: the same argv passed to the trainer, so `--agent <agent>` seeds <agent>,
# while a run that names no agent seeds the GUI-selected agent.
run_seed() {
    if [[ -z "${R2_BUCKET:-}" ]]; then
        echo "[entrypoint] R2_BUCKET not set — skipping checkpoint seed."
        return 0
    fi
    local seed_agent
    seed_agent="$(resolve_agent_arg "$@")"

    # `models-path` only needs the agent name, not settings content, so it is safe to call
    # before the remote forge_settings.json has landed.
    local models_rel settings_rel
    models_rel="$(ForgeConsoleTrainer models-path ${seed_agent:+--agent "$seed_agent"})" || models_rel=""
    if [[ -n "$models_rel" ]]; then
        settings_rel="${models_rel%/Models}/forge_settings.json"
        rclone copyto "r2:${R2_BUCKET}/${settings_rel}" "${SRC_ROOT}/${settings_rel}" --update \
            || echo "[entrypoint] (no remote forge_settings.json yet — using baked-in defaults)"
    fi

    local mode label filter
    for mode in --metadata-only ""; do
        if ! filter="$(write_manifest "$seed_agent" "$mode")"; then
            echo "[entrypoint] WARNING: could not compute the sync manifest — skipping seed."
            return 0
        fi
        if [[ -n "$mode" ]]; then label="build metadata"; else label="training manifest"; fi
        echo "[entrypoint] Seeding ${label} from r2:${R2_BUCKET}/Bots ..."
        mkdir -p "${SRC_ROOT}/Bots"
        rclone copy "r2:${R2_BUCKET}/Bots" "${SRC_ROOT}/Bots" --filter-from "$filter" "${RCLONE_SYNC_FLAGS[@]}" \
            || echo "[entrypoint] WARNING: seed failed — training may start from scratch."
        rm -f "$filter"
    done
}

# Starts sshd if the host injected an authorized key. Entrypoint mode replaces
# Vast.ai's/RunPod's own SSH-setup script (see Documentation/RemoteTraining/VastAI.md),
# so without this nothing in the container ever listens on port 22. Both hosts use
# the same $PUBLIC_KEY convention for this; no-op (and no sshd) when it's unset, e.g.
# a local `docker run`.
start_sshd() {
    if [[ -z "${PUBLIC_KEY:-}" ]]; then
        return 0
    fi
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    echo "[entrypoint] PUBLIC_KEY set — starting sshd."
    /usr/sbin/sshd
}

# True when running on a recognized self-terminating cloud host. RunPod injects
# RUNPOD_POD_ID; Vast.ai injects CONTAINER_ID + CONTAINER_API_KEY (per-instance
# scoped key). Keyed on CONTAINER_API_KEY for Vast because CONTAINER_ID alone is
# too generic to be a reliable signal.
on_cloud_host() {
    [[ -n "${RUNPOD_POD_ID:-}" ]] && return 0
    [[ -n "${CONTAINER_ID:-}" && -n "${CONTAINER_API_KEY:-}" ]] && return 0
    return 1
}

# Destroy the current cloud instance to stop billing, selecting the host's CLI.
# Only called after a verified export. A missing/failed CLI warns and leaves the
# instance up for manual termination (never silently strands the meter running
# without telling the operator).
self_terminate() {
    if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
        echo "[entrypoint] Export verified. Self-terminating RunPod pod ${RUNPOD_POD_ID}."
        if command -v runpodctl >/dev/null 2>&1; then
            runpodctl remove pod "$RUNPOD_POD_ID" \
                || echo "[entrypoint] WARNING: self-terminate failed — terminate manually."
        else
            echo "[entrypoint] WARNING: runpodctl not found — cannot self-terminate. Terminate manually."
        fi
    elif [[ -n "${CONTAINER_ID:-}" && -n "${CONTAINER_API_KEY:-}" ]]; then
        echo "[entrypoint] Export verified. Self-terminating Vast.ai instance ${CONTAINER_ID}."
        if command -v vastai >/dev/null 2>&1; then
            vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" --yes \
                || echo "[entrypoint] WARNING: self-terminate failed — terminate manually."
        else
            echo "[entrypoint] WARNING: vastai not found — cannot self-terminate. Terminate manually."
        fi
    else
        echo "[entrypoint] WARNING: no recognized cloud host — terminate manually."
    fi
}

# Periodically export the manifest's files to R2 while the trainer runs, so a lost instance
# costs at most one interval of progress. Reuses run_export verbatim, so a backup
# and the final export are byte-for-byte the same operation. Disabled when R2 is
# unconfigured or BACKUP_INTERVAL_SECONDS<=0. A failed cycle warns and retries;
# it never idles or self-terminates (only the FINAL export gates lifecycle).
backup_loop() {
    local agent="${1:-}"
    local interval="${BACKUP_INTERVAL_SECONDS:-600}"
    # Disabled when R2 is unconfigured, or when the interval is not a positive
    # integer (treats a typo like "10m" as "off" instead of erroring on the
    # arithmetic test below).
    if [[ -z "${R2_BUCKET:-}" ]] || ! [[ "$interval" =~ ^[0-9]+$ ]] || [[ "$interval" -le 0 ]]; then
        return 0
    fi
    while true; do
        sleep "$interval"
        echo "[entrypoint] Periodic backup to R2 ..."
        run_export "$agent" || echo "[entrypoint] WARNING: periodic backup had failures — retrying next interval."
        run_export_log   # mirror the final path: also push the latest log snapshot
    done
}

main() {
    # Capture the entire entrypoint's output (trainer + [entrypoint] lines) to a
    # container-local log while still streaming live to the console. `tee` (no -a)
    # truncates LOG_FILE, so each run starts a fresh log. This does NOT disturb
    # trainer_pid below (process substitution and `exec` do not set $!; only the
    # backgrounded trainer does): the trainer is still a normal background job.
    exec > >(tee "$LOG_FILE") 2>&1

    # The seed/export agent, derived the same way run_seed does.
    seed_agent="$(resolve_agent_arg "$@")"

    start_sshd

    # With no start command the trainer itself defaults to the saved queue for the
    # GUI-selected agent, so no command injection is needed here.
    run_seed "$@"
    run_seed_newest_temp_checkpoints "$seed_agent"
    run_prune_temp_checkpoints "$seed_agent"

    # Forward graceful-stop signals so the trainer saves a checkpoint before exiting,
    # and USR1 (`docker kill -s USR1`) so the trainer skips to its next queued stage.
    # Installed before the launch so no signal falls in a gap. A stop that arrives before
    # trainer_pid is set is held and forwarded once, since a second TERM kills the trainer outright.
    stop_pending=0
    forward_term() {
        if [[ -n "${trainer_pid:-}" ]]; then
            kill -TERM "$trainer_pid" 2>/dev/null || true
        else
            stop_pending=1
        fi
    }
    trap forward_term TERM INT
    forward_usr1() {
        [[ -n "${trainer_pid:-}" ]] && kill -USR1 "$trainer_pid" 2>/dev/null
        return 0
    }
    trap forward_usr1 USR1

    # Launch the trainer in the background so we can trap signals and export after it exits.
    ForgeConsoleTrainer "$@" &
    trainer_pid=$!
    if [[ $stop_pending -eq 1 ]]; then kill -TERM "$trainer_pid" 2>/dev/null || true; fi

    # Periodically back up the manifest's files to R2 while training runs (no-op
    # without R2 or with BACKUP_INTERVAL_SECONDS<=0). Killed before the final export
    # below so the two never run concurrently.
    backup_loop "$seed_agent" &
    backup_pid=$!

    # The first wait is interrupted when a trapped signal fires; loop with kill -0
    # until the trainer process is actually gone, capturing its real exit code.
    # A forwarded USR1 leaves the trainer running, so the loop simply waits again.
    wait "$trainer_pid"; trainer_exit=$?
    while kill -0 "$trainer_pid" 2>/dev/null; do
        wait "$trainer_pid"; trainer_exit=$?
    done

    # Stop the periodic backup so it cannot race the authoritative final export.
    kill "$backup_pid" 2>/dev/null || true
    wait "$backup_pid" 2>/dev/null || true

    run_export "$seed_agent"; export_ok=$?

    # Upload the run log too (non-critical — never folded into export_ok).
    run_export_log

    # Self-manage lifecycle ONLY on a recognized cloud host that has R2 configured.
    # Every other environment (local docker, other hosts) falls through to a normal exit.
    if [[ -n "${R2_BUCKET:-}" ]] && on_cloud_host; then
        if [[ $export_ok -ne 0 ]]; then
            echo "[entrypoint] Export FAILED — idling so you can recover checkpoints. Terminate manually."
            sleep infinity
        else
            self_terminate
        fi
    fi

    exit "$trainer_exit"
}

# Only run main when executed directly; sourcing (e.g. tests) just loads the helpers.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
