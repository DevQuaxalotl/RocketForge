# Remote training on Vast.ai

This guide shows you how to run the RocketForge training image on Vast.ai, which we recommend.
Before you start, complete the common setup in [Remote training](RemoteTraining.md):
Docker Hub, Cloudflare R2, W&B, the local environment file, the smoke test, and the push and pull workflow.

We recommend Vast.ai because training throughput is bound by *collection*,
which is RocketSim environment stepping on the CPU.
Collection is bound by per-core clock speed, not by the vCPU count or the GPU tier.
Vast.ai's marketplace lets you filter for **high-clock CPUs**,
which is often the biggest single lever on steps per second.
For more information, see [Performance expectations](RemoteTraining.md#performance-expectations).

## Set up Vast.ai

1. Create an account at <https://vast.ai>, and add credit.
1. Optional: In **Account > API Key**, note your account API key.

   The key doesn't go in the container, because Vast.ai gives each instance its own key.
   For more information, see [Self-termination](#self-termination).
   You need the account key only for the `vastai` CLI on your own machine, to manage instances from your desktop.
1. Optional: To get SSH access to your instances, add an SSH key in **Account > SSH Keys** before you deploy.
   For more information, see [Connect over SSH](#connect-over-ssh).

Vast.ai has no secrets vault, unlike RunPod.
Instead, you pass credentials as Docker `-e` environment variables on the instance.

## Deploy and run

1. Create a template, or edit the settings at deploy time:

   - **Image:** `docker.io/DOCKERHUB_USER/rocketforge:latest`, or the tag that you pushed with `tools/remote/push_training_image.ps1`.
   - **Launch mode:** **Entrypoint**.

     This setting is critical.
     The **SSH** and **Jupyter** launch modes replace the image's entrypoint with their own setup script,
     so the trainer never runs.
     Only **Entrypoint** mode runs the image's own `ENTRYPOINT`, the `entrypoint.sh` script.
     If your template defaults to **SSH** or **Jupyter**, switch it to **Entrypoint**.
   - **Disk:** at least **40 GB**, for libtorch, build artifacts, and checkpoints.
   - **Docker options:** pass each credential that the entrypoint needs with `-e`:

     ```
     -e WANDB_API_KEY=your_wandb_key
     -e R2_BUCKET=rocketforge-checkpoints
     -e RCLONE_CONFIG_R2_TYPE=s3
     -e RCLONE_CONFIG_R2_PROVIDER=Cloudflare
     -e RCLONE_CONFIG_R2_REGION=auto
     -e RCLONE_CONFIG_R2_ACCESS_KEY_ID=your_access_key_id
     -e RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=your_secret_access_key
     -e RCLONE_CONFIG_R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com
     ```

     These are the same values as in your `~/.rocketforge_env` file.
     Optional: To tune the periodic backup, add `-e BACKUP_INTERVAL_SECONDS=600`.
     For more information, see [Periodic backup during a run](RemoteTraining.md#periodic-backup-during-a-run).
   - **Registry credentials:** if your Docker Hub repository is private,
     add your Docker Hub username and access token, so that Vast.ai can pull the image.
   - **On-start command:** leave it empty.
     The trainer defaults to the saved stage queue of the GUI-selected agent, baked into `forge_app_state.json`.

1. Filter the GPU offers for a high-clock CPU and an adequate GPU, such as an RTX 4090, and click **Rent**.
1. Monitor the run on W&B.

   The instance logs print the run URL,
   followed by `[entrypoint] Seeding build metadata from r2:...` and `[entrypoint] Seeding training manifest from r2:...`,
   so that you can confirm that the run continued the right stage.
   The periodic backup prints `[entrypoint] Periodic backup to R2 ...` at every interval.

1. To stop the run, click the instance's **Stop** button in the Vast.ai console.

   You don't need a shell.
   **Stop** runs `docker stop`, which sends `SIGTERM` to the entrypoint, process 1,
   and starts the same save, R2 export, and self-termination sequence as `stop-training`.
   A bounded queue also stops on its own.

### Connect over SSH

As an alternative to the **Stop** button, you can connect over SSH and run `stop-training`.
To skip the current stage and continue with the next queued one, run `kill -USR1 1` instead.

Entrypoint mode replaces Vast.ai's own SSH setup script,
so `sshd` starts only if you add an SSH key in **Account > SSH Keys** before you deploy.
Vast.ai then passes the key to the instance as `$PUBLIC_KEY`,
and the `start_sshd` function in `entrypoint.sh` writes it to `authorized_keys` and launches `sshd`.

An instance deployed without a registered key has no SSH access at all:
`ssh_host` and `ssh_port` are `null` in `vastai show instance --raw`.
Adding a key afterward doesn't start `sshd` on a running instance,
so redeploy after you register the key.

To find the connect command, use one of the following:

- The instance's detail page, which shows `ssh -p PORT root@HOST`.
- The CLI command `vastai ssh-url INSTANCE_ID`.

  The `vastai` package on PyPI can lag behind Vast.ai's API.
  If `ssh-url` fails with a `410` or deprecated-endpoint message,
  download the current script from `https://raw.githubusercontent.com/vast-ai/vast-cli/master/vast.py`.

## Self-termination

Vast.ai sets the following two environment variables in every instance:

- `CONTAINER_ID`: the instance ID.
- `CONTAINER_API_KEY`: a **per-instance scoped** API key for CLI commands from inside the container.

After a **verified** export, the entrypoint runs
`vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" --yes`,
and the instance self-terminates, so billing stops with no idle time.
Because the key is scoped to the instance, **no account-wide secret goes in the container.**

If the export **fails**, the entrypoint doesn't self-terminate.
It idles with an `Export FAILED — idling` banner, which protects the checkpoints on disk.
Fix the credentials, run the export again, and then destroy the instance manually.
**Never destroy the instance first.**

## Troubleshooting

| Symptom | Fix |
|---|---|
| The trainer never starts, or only an SSH or Jupyter session appears. | The instance launched in **SSH** or **Jupyter** mode, which replaces the entrypoint. Redeploy in **Entrypoint** launch mode. |
| `[entrypoint] R2_BUCKET not set` | The `-e R2_BUCKET` and `-e RCLONE_CONFIG_R2_*` Docker options aren't set on the instance. Add them to the template's **Docker options** field. |
| The instance didn't self-terminate. | `vastai` is missing, or `CONTAINER_ID` or `CONTAINER_API_KEY` isn't set, which can happen on older instances. Alternatively, the export failed and the instance is idling on purpose. Destroy the instance manually. |
| No W&B run appears. | `-e WANDB_API_KEY` is missing or mistyped in the Docker options. |
| The image doesn't pull from a private repository. | Add your Docker Hub username and access token to the instance's registry credentials. |
