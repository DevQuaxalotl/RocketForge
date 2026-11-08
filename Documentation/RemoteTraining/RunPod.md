# Remote training on RunPod

This guide shows you how to run the RocketForge training image on RunPod.
Before you start, complete the common setup in [Remote training](RemoteTraining.md):
Docker Hub, Cloudflare R2, W&B, the local environment file, the smoke test, and the push and pull workflow.

## Set up RunPod

1. Create an account at <https://runpod.io>, and add credit.
1. Go to **Settings > Container Registry Auth**, and add your Docker Hub username and access token.
   This lets pods pull your private image.
1. Go to **Settings > Secrets**, and create one secret for each credential that the pod needs:
   `WANDB_API_KEY`, `R2_BUCKET`, and each `RCLONE_CONFIG_R2_*` value from your environment file.
   Secrets keep credentials out of the template definition.
1. Go to **Templates > New Template**, and create a template named `rocketforge` with the following settings:

   - **Container Image:** `docker.io/DOCKERHUB_USER/rocketforge:latest`
   - **Registry Credentials:** the registry authentication that you added.
   - **Container Disk:** **40 GB**, for libtorch, build artifacts, and checkpoints.
   - **Environment Variables:** connect each variable to its matching secret.
   - **Start Command:** leave it empty.
     The trainer defaults to the saved training queue of the agent baked into `forge_app_state.json`.

## Deploy and run

1. Click **Deploy**, and choose the `rocketforge` template.
1. Pick a GPU. An RTX 4090 is a good default.
1. Set the image tag to the tag that you pushed, or to `latest`.
1. Leave the start command empty, and deploy.

   The pod seeds checkpoints from R2 and then starts training.
1. Monitor the run on W&B.

   The pod logs print the run URL.
   Open it at wandb.ai, and watch the reward curves and the steps per second.
   To confirm that the run continued your segment,
   look for `[entrypoint] Seeding build metadata from r2:...`
   followed by `[entrypoint] Seeding training manifest from r2:...` in the seed step.
1. Stop the run, or let a bounded queue finish on its own.

   To stop early, open the pod's web terminal,
   or connect over SSH from the pod's detail page (**Connect > SSH over exposed TCP**),
   and run the following command, which is an alias for `kill -TERM 1`:

   ```bash
   stop-training
   ```

   To skip the current stage and continue with the next queued one, run `kill -USR1 1` instead.

## Self-termination

When the pod stops, the entrypoint saves the final checkpoint and exports the queue's files to R2.
What happens next depends on the export:

- **The export succeeds:** the entrypoint calls `runpodctl remove pod $RUNPOD_POD_ID`,
  and the pod self-terminates, so billing stops with no idle time.
  Look for `[entrypoint] Export verified. Self-terminating RunPod pod ...`.
- **The export fails:** the pod idles with an `Export FAILED — idling` banner.
  Idling protects the checkpoints from being destroyed with the disk.
  Fix the credentials, run the export again, and then terminate the pod manually.

Self-termination happens only on a RunPod pod with R2 configured.

## Troubleshooting

| Symptom | Fix |
|---|---|
| The pod is stuck pulling, or shows an authentication error. | Check the RunPod **Container Registry Auth** username and token, and confirm that the image tag exists. |
| The pod didn't self-terminate. | `runpodctl` is missing, or `RUNPOD_POD_ID` isn't set. Terminate the pod manually. |
| The pod exits immediately, and the UI shows **No such container**. | The trainer stopped at startup, so the container exited and the pod self-terminated; the message means that the container is already gone. Check the pod log for `No version metadata or checkpoints found ... Use StartNewStage`, which means that your `continueExisting` queue's checkpoints weren't seeded. Run `push_models.ps1` so that R2 has them, or make the first stage a new one. A `Seeding Bots/.../Models` line means that the image is stale; rebuild it. |
