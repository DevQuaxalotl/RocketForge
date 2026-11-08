# Remote training

This guide shows you how to train a RocketForge agent on a rented cloud GPU, from start to finish.
It doesn't assume that you've used Docker, Cloudflare R2, or a GPU cloud before.

## How remote training works

You build the trainer once on your machine, into a *Docker image*:
a self-contained Linux environment with the compiled `ForgeConsoleTrainer`, libtorch, and your bot source.
You push that image to a private registry,
and then you rent a GPU instance on Vast.ai or RunPod that boots from it.
The instance (also called a *pod*) pulls your latest checkpoints, trains, uploads the results, and shuts itself off.
You then pull the results back and keep going.

```
  YOUR PC                         CLOUD                         YOUR PC
  ┌──────────────┐                                             ┌──────────────┐
  │ build image  │── docker push ─▶ Docker Hub ─┐              │ pull_models  │
  │ push models  │── rclone ──────▶ R2 ─────┐   │              │  (rclone)    │
  └──────────────┘                          │   │              └──────▲───────┘
                                            ▼   ▼                     │
                                   GPU instance boots from image      │
                                   1. seed checkpoints  ◀── R2        │
                                   2. train (watch on W&B)            │
                                   3. export checkpoints ── R2 ───────┘
                                   4. self-terminate (billing stops)
```

The workflow uses the following four external services, each of which you set up once:

| Service | Role | Cost |
|---|---|---|
| **Docker Hub** | Stores the built training image so that the GPU host can pull it. | Free for one private repository. |
| **Cloudflare R2** | Stores training checkpoints, in both directions. | Free tier: 10 GB, no egress fees. |
| **Vast.ai** or **RunPod** | Rents the GPU that runs the image. We recommend Vast.ai; for more information, see [Choose your GPU host](#choose-your-gpu-host). | Billed per hour, for example about $0.34 per hour for an RTX 4090. |
| **Weights & Biases (W&B)** | Live training metrics dashboard. | Free for personal use. |

Checkpoints move through R2 with `rclone copy --update`,
which **never deletes** files and **never overwrites a newer file**,
so you can safely repeat any pull or push.

### What each sync transfers

Every training sync point transfers only what the selected agent's stage queue touches,
as listed by `ForgeConsoleTrainer sync-manifest`.
The sync points are `push_models.ps1`, `pull_models.ps1 -Agent -TrainingOnly`,
and the pod's seed, export, and periodic backup.
The manifest lists the following items:

- The build's metadata: `forge_settings.json`, `builds.json`, `build.json`, the registries,
  and every segment's `segment.json` and `training_setup.json` files.
- The **whole folders** of the segments that the queue continues, loads from, or created.
- The recent-policy and skill-benchmark pool entries whose source segment is in that lineage.
- The static opponents that the build names, both imported models and source segments.

Each sync point writes that list to a temporary file and runs **one** `rclone copy --filter-from` call rooted at `Bots/`,
so rclone never walks the segments and pool entries that it doesn't need.
On a build with dozens of segments, that's the difference between minutes and seconds.

The `forge_settings.json` file is baked into the image, but it also travels through R2,
where the last writer wins through `--update`.
It travels because it carries `createdSegmentId`, which records the segment that a run already created or continued.
Pulling it first, with the settings-independent `models-path` command,
lets the next run, local or remote, continue that segment instead of creating a duplicate.
Pulls, meaning `-TrainingOnly` and the pod's seed, then run a `--metadata-only` pass before the full pass,
because the lineage walk reads `segment.json` files that a remote run might have created.

The `forge_app_state.json` file, which records the selected agent, is only baked into the image and never travels through R2.
`build-info` (`BUILD_FOLDER`, `MODELS_REL`, `START_SEGMENT`, `TAG`, and the other keys) names the image tag.

> **Important:** Never train the same stage in two places at the same time.
> `--update` resolves conflicts by keeping the newest file,
> and two divergent runs of the same stage can't be merged.

## Choose your GPU host

The common setup on this page (Docker Hub, R2, W&B, the smoke test, and pushing and pulling) is the same for every host.
Deployment and self-termination differ by host.
After you finish [Set up the services](#set-up-the-services), follow one of these guides:

| Host | Guide | Notes |
|---|---|---|
| **Vast.ai** (recommended) | [Remote training on Vast.ai](VastAI.md) | The marketplace lets you pick **high-clock CPUs**, which directly raises the collection-bound steps per second. Each instance gets its own API key, so no account secret goes in the container. |
| **RunPod** | [Remote training on RunPod](RunPod.md) | Uses a secrets vault and template environment variables. Self-terminates with `runpodctl`. |

We recommend Vast.ai because throughput is bound by CPU clock speed,
and Vast.ai lets you select faster cores.
For more information, see [Performance expectations](#performance-expectations).

## Set up the services

You do this section once.
Afterward, every run is a two-command push plus a few clicks.

### Install local tools

- **Docker Desktop**, which builds the image.
  Download it from <https://www.docker.com/products/docker-desktop/>.
  BuildKit, which makes rebuilds fast, is on by default.
- **rclone**, which pushes and pulls checkpoints to R2.
  Install it with `winget install Rclone.Rclone`, or download it from <https://rclone.org/downloads/>.
  To verify the installation, run `rclone version`.

### Set up Docker Hub

1. Create a free account at <https://hub.docker.com>.
1. Create a **private** repository named `rocketforge`.
1. Go to **Account Settings > Security > New Access Token**, and create a token with the **Read & Write** scope.
   Copy the token.
1. Sign in locally, so that `docker push` is authorized.
   When prompted for a password, paste the access token.

   ```powershell
   docker login -u DOCKERHUB_USER
   ```

   Replace `DOCKERHUB_USER` with your Docker Hub username.

### Set up Cloudflare R2

1. Sign in at <https://dash.cloudflare.com>, or create an account.
1. In the sidebar, click **R2**.

   The first time, Cloudflare asks you to add a payment method to turn on R2.
   The payment method is required even though the free tier doesn't charge you until you exceed its limits.
   Optional: set a billing alert.
1. Click **Create bucket**, name the bucket `rocketforge-checkpoints`, keep the default location, and create the bucket.
1. Click **Manage R2 API Tokens > Create API Token**.
   Give the token the **Object Read & Write** permission, scope it to that bucket or to all buckets, and create it.
1. Copy the following three values now, because Cloudflare shows them only once:

   - **Access Key ID**
   - **Secret Access Key**
   - **S3 endpoint**, in the form `https://ACCOUNT_ID.r2.cloudflarestorage.com`.
     The bucket's **Settings** page also shows it.

### Get a Weights & Biases API key

W&B is how you watch a run.
Copy your API key from <https://wandb.ai/authorize>.

### Create the local environment file

rclone reads a remote named `r2` directly from environment variables, so you don't need to run `rclone config`.

1. Put every credential in the `~/.rocketforge_env` file.
   Save it as ASCII, because a UTF-8 byte order mark can corrupt the first key.

   ```powershell
   @"
   WANDB_API_KEY=your_wandb_key
   R2_BUCKET=rocketforge-checkpoints
   RCLONE_CONFIG_R2_TYPE=s3
   RCLONE_CONFIG_R2_PROVIDER=Cloudflare
   RCLONE_CONFIG_R2_REGION=auto
   RCLONE_CONFIG_R2_ACCESS_KEY_ID=your_access_key_id
   RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=your_secret_access_key
   RCLONE_CONFIG_R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com
   RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true
   "@ | Set-Content -Encoding ascii "$env:USERPROFILE\.rocketforge_env"
   ```

   `RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true` is required because the R2 token is scoped to one bucket and **can't create buckets**.
   Without it, rclone runs its default "ensure the bucket exists" step before every upload,
   which calls `CreateBucket`.
   R2 rejects that call with `403 AccessDenied`, and the push or export stops.
   The bucket already exists, so skipping the check is safe.
   The container sets this variable itself, so a missing pod variable doesn't break the export,
   but keep it in this file so that a local `push_models.ps1` works.

1. Before you spend anything on a GPU, verify the keys:

   ```powershell
   Get-Content "$env:USERPROFILE\.rocketforge_env" | ForEach-Object { if ($_ -match '^\s*([^#=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process') } }
   rclone lsf r2:rocketforge-checkpoints
   ```

   The command succeeds with exit code 0 and lists the bucket's contents,
   which is empty output for a new bucket.
   An authentication error means a bad key or endpoint.
   Fix it now, before you're paying for a GPU.

   Don't use `rclone lsd r2:` for this check.
   It lists all buckets, which needs account-level permission,
   so a valid token scoped only to `rocketforge-checkpoints` fails it.
   `rclone lsf r2:rocketforge-checkpoints` reads inside the bucket, which the scoped token can do.

### Set up your GPU host

To set up your host account, follow the guide for [Vast.ai](VastAI.md) (recommended) or [RunPod](RunPod.md).
Then return here for the smoke test.

## Smoke-test the image locally

Before you rent a GPU, confirm that the image builds and that the trainer runs on your own machine.
Catching a broken build or a bad `forge_*.json` file here costs nothing;
catching it on a running pod costs GPU time and a round trip.
Everything in this section runs against local Docker only, with no Docker Hub push and no pod.

The training run in this section needs an NVIDIA GPU exposed to Docker.
On Windows, that means Docker Desktop's WSL 2 backend, an NVIDIA card, and current drivers, so that `--gpus all` works.
Without an NVIDIA GPU, you can still run the build and the link check, which don't need a GPU.

### Build the image

The push script builds the image before it pushes it,
so a local `docker build` exercises the same path without touching Docker Hub or R2.

1. Check out the RocketSim v3 source:

   ```powershell
   git submodule update --init ForgeTraining/RLGym/RocketSimV3/upstream
   ```

   `.dockerignore` excludes every `.git` folder, so the build can't fetch the submodule itself.
   Without the submodule, the build stops at the configure step with
   `RocketSim v3 isn't built: its source (the ForgeTraining/RLGym/RocketSimV3/upstream submodule) is missing`.
   `push_training_image.ps1` runs the same command before it builds.

1. Build the image:

   ```powershell
   docker build -t rocketforge .
   ```

   The first build takes about 20-30 minutes, because it downloads libtorch and the Rust toolchain and compiles everything.
   Later builds are incremental.
   A successful build means that the Linux toolchain, CUDA detection, and your bot source all compile,
   which rules out the most common cause of a wasted pod.

The image contains the following:

- `ForgeConsoleTrainer` with **both engines**, RocketSim v2 and v3.
  The image configures with `-DROCKETFORGE_ENABLE_V3=ON`, so it never silently ships v2 only,
  and the build log shows `Engines: v2 (built), v3 (built)`.
- libtorch 2.13.0+cu129 on CUDA 12.9,
  and Rust 1.99.0, the toolchain pinned in `ForgeTraining/RLGym/RocketSimV3/bindings/rust-toolchain.toml`.
- The full source tree, including the `Bots/` source and settings and the collision meshes.
  It excludes `Bots/**/Models`, which comes from R2 or a volume mount.
- `wandb`, `rclone`, `runpodctl`, and `vastai`, for metrics, checkpoint sync, and pod self-termination.

#### Optional: Run the unit tests in the image

To run the unit tests for both engines inside the image,
start a shell in the image and run the Linux build script:

```powershell
docker run --rm -it --gpus all `
  -v ${PWD}/Bots/PublicBots:/workspace/RocketForge/Bots/PublicBots:ro `
  --entrypoint bash rocketforge -c "tools/build/build_and_test.sh"
```

The script rebuilds from scratch inside the container, because the image keeps only the trainer binary, not its build tree.
The read-only `Bots/PublicBots` mount restores the public bots' models,
which the image excludes along with every other `Models/` folder;
without the mount, the PublicBots and static-opponent tests fail.

On Linux, the Qt-only tests report as skipped, because the image has no Qt.
One failure is known on Linux:
`PublicBots.PaddedAttentionPoliciesMatchPublishedVariablePlayerOutputs` exceeds its CPU logit tolerance.

### Check that the binary links

To confirm that the compiled binary loads and links against libtorch,
run the build-path resolver that `push_training_image.ps1` uses to name the image tag.
It's a lightweight, CPU-only subcommand that exits immediately, so it doesn't need a GPU.

```powershell
docker run --rm rocketforge build-info
```

The command prints the selected build's `KEY=VALUE` lines and exits with code 0.
The lines include `BUILD_FOLDER`, `MODELS_REL` (for example, `Bots/.../Models`), `START_SEGMENT`, and `TAG`.
An error means that the binary or its `.so` dependencies are broken; fix it before you continue.

Because `R2_BUCKET` isn't set, the output also includes
`[entrypoint] R2_BUCKET not set — skipping checkpoint seed`,
which confirms that the path without cloud storage works.

### Run a training smoke test

This test needs a GPU.

1. Run a short stage with metrics turned off,
   and mount a host folder so that the checkpoints it writes land on your machine:

   ```powershell
   docker run --gpus all `
     -v ${PWD}/Bots:/workspace/RocketForge/Bots `
     rocketforge `
     train --agent example-agent --stage example-ball-chase --no-metrics
   ```

   - `--no-metrics` skips W&B, so the smoke test doesn't need an API key.
   - The `-v` mount keeps `Bots/**/Models` on your machine.
     Without it, the checkpoints disappear when Docker removes the container.
   - To use your own agent and stage, replace `--agent example-agent --stage example-ball-chase`.
     To run the GUI-selected queue, which is what a pod runs by default, omit the trailing arguments.

1. Let it train for a few iterations.
1. Stop the container to exercise the graceful save and the skipped export.
   Press **Control+C** in the foreground, or run the following command from another terminal:

   ```powershell
   docker stop -t 60 CONTAINER
   ```

   Replace `CONTAINER` with the container name that `docker ps` shows.

   The default `docker stop` grace period is 10 seconds, which can send `SIGKILL` to the trainer in the middle of a save.
   `-t 60`, or **Control+C**, which has no timeout, lets the checkpoint finish writing.
   Pressing **Control+C** a second time terminates the trainer immediately, so wait for the save to finish.

   To exercise the stage skip, run `docker kill -s USR1 CONTAINER` from another terminal.
   The trainer saves and moves on to the next queued stage.

The trainer saves a final checkpoint,
prints `[entrypoint] R2_BUCKET not set — skipping checkpoint export`,
and exits cleanly without self-terminating, because `RUNPOD_POD_ID` isn't set locally.
That's the cloud sequence without the R2 upload.
If it works here, then the only new variables on the pod are the GPU type and the R2 credentials.

### Rehearse a full pod run

The smoke test stays offline.
To rehearse exactly what a pod does,
run the image from [Build the image](#build-the-image) with your environment file.
It's the same image, default command, and credentials that a pod uses:

```powershell
docker run --gpus all --env-file $env:USERPROFILE\.rocketforge_env rocketforge
```

This is the same `rocketforge` image that `push_training_image.ps1` builds and pushes to Docker Hub
in [Run your first remote training](#run-your-first-remote-training).

The run goes through the pod's lifecycle:

1. **Seed.** `rclone copy` pulls what the GUI-selected agent's stage queue needs from R2 into the container.
   The image ships without `Models`, exactly like the pod's disk.
   Look for `[entrypoint] Seeding build metadata from r2:...`,
   followed by `[entrypoint] Seeding training manifest from r2:...`.

   The seed must succeed for a `continueExisting` queue.
   A saved stage queue usually continues an existing version,
   so the trainer needs that stage's checkpoints after the seed.
   If they aren't in R2 yet, the trainer stops immediately with
   `No version metadata or checkpoints found ... Use StartNewStage`.
   That stops the container, so a RunPod pod then self-terminates,
   and the web UI shows a confusing **No such container** message.
   Only a queue whose first stage is a new stage (`StartNewStage`) starts from scratch on an empty seed.
   So before the first pod run, push your models
   (`push_training_image.ps1` and `./tools/remote/push_models.ps1` both upload them),
   or make the first stage a new one.

1. **Train.** Because no command is given, the trainer runs the saved stage queue
   and logs a real W&B run, using the `WANDB_API_KEY` from the environment file.

1. **Export.** Stop the run with `docker stop -t 60 CONTAINER`, `stop-training` from inside the container, or **Control+C**.
   The entrypoint saves the final checkpoint,
   and then copies the queue's segments, pools, and metadata back up to R2 with `rclone copy`.
   Look for `[entrypoint] Exporting training manifest to r2:...`.

The only difference from a pod is the last step:
because `RUNPOD_POD_ID` isn't set, the entrypoint exits cleanly instead of self-terminating.
To confirm the upload, run `rclone lsf r2:rocketforge-checkpoints`,
or pull the results back with `./tools/remote/pull_models.ps1`.

This rehearsal deliberately has no volume mount, because a pod doesn't mount anything either;
it works entirely from R2.
To also keep the run's output on your local disk, add `-v ${PWD}/Bots:/workspace/RocketForge/Bots`,
but then the run no longer matches a pod exactly.

> **Important:** This rehearsal is a live run against the real bucket.
> It seeds from and exports to R2 and opens a real W&B run.
> Don't train the same stage here and on a pod at the same time,
> and remember that the checkpoints it produces are now in R2.

When all of these checks pass, you're ready to train remotely.

## Run your first remote training

1. In the GUI, pick the agent and the training settings.

   Selecting them writes the `forge_app_state.json` file (the selected agent)
   and the `forge_settings.json` file (the stage queue).
   Both are baked into the image, so the pod runs exactly what you selected.

1. Build the image, push it, and seed R2 with one command:

   ```powershell
   $env:DOCKERHUB_USER = "DOCKERHUB_USER"
   ./tools/remote/push_training_image.ps1 -Tag first-run
   ```

   Replace `DOCKERHUB_USER` with your Docker Hub username.
   You need to set it once for each shell, or set it persistently.

   The script builds the image, pushes the `:<tag>` and `:latest` tags to Docker Hub,
   and then runs `push_models.ps1`,
   so that R2 has what the selected agent's stage queue needs and the pod continues from where you are.
   For more information, see [What each sync transfers](#what-each-sync-transfers).

   `-Build` overrides only which build is pushed to R2.
   It doesn't change the build that the image trains,
   which still comes from the `forge_settings.json` file baked into the image.
   For normal cloud training, make sure that `-Build` matches the GUI-selected build.

   The first build takes about 20-30 minutes, because it downloads about 2 GB of libtorch and compiles everything.
   Later builds are incremental and usually take under 2 minutes.

   If you omit `-Tag`, the image tag defaults to `<BuildFolder>-<StartSegment>`, for example `ExampleBuild-v1.01`,
   derived from `ForgeConsoleTrainer build-info`.
   That way, the image name records which build and segment it starts training.
   If the console trainer isn't built, the tag falls back to a `yyyyMMdd-HHmm` timestamp.
   An explicit `-Tag` always takes precedence.

1. To deploy the image that you pushed and start training,
   follow the guide for [Vast.ai](VastAI.md) or [RunPod](RunPod.md).

1. Monitor the run on W&B.

   The pod's logs print the run URL.
   Open it at wandb.ai, and watch the reward curves and the steps per second.
   To confirm that the run continued your segment,
   look for `[entrypoint] Seeding build metadata from r2:...`
   followed by `[entrypoint] Seeding training manifest from r2:...` in the seed step.

1. Stop the run, or let a bounded queue finish on its own.

   To stop early, connect through the instance's web terminal or SSH, and run the following command,
   which is an alias for `kill -TERM 1`:

   ```bash
   stop-training
   ```

   Either way, the entrypoint saves the final checkpoint
   and exports the queue's segments, pools, and metadata to R2.
   Self-termination after a verified export depends on the host;
   for more information, see your host's guide.

   To skip the current stage instead of ending the run, run `kill -USR1 1` in the pod's terminal,
   or `docker kill -s USR1 CONTAINER` where you have Docker.
   The trainer saves, then continues with the next queued stage.
   A second **Control+C** or `SIGTERM` while the trainer is saving terminates it immediately, so send a stop only once.

1. Pull the results back:

   ```powershell
   ./tools/remote/pull_models.ps1 -Agent example-agent -TrainingOnly
   ```

   The script loads the R2 credentials from `~/.rocketforge_env` and runs `rclone copy --update`.
   The checkpoint format is identical, so you can resume training locally immediately.

### Pull options

`pull_models.ps1` accepts the following forms:

```powershell
./tools/remote/pull_models.ps1 -Agent example-agent -TrainingOnly   # what the queue touched (fast)
./tools/remote/pull_models.ps1 -Agent example-agent                 # the agent's whole Models tree
./tools/remote/pull_models.ps1                                    # whole Bots tree
./tools/remote/pull_models.ps1 -Agent example-agent -Build MyBuild
```

- `-Agent AGENT -TrainingOnly` is the normal pull after a run. `-TrainingOnly` requires `-Agent`.
  It fetches `forge_settings.json`, then the build metadata,
  and then only the segments and pool entries that the queue touches.

  If your local `forge_settings.json` file is newer than the remote one, the script keeps it,
  but gives it the segment IDs that the remote run recorded (`ForgeConsoleTrainer merge-segment-ids`),
  so that the local queue continues those segments instead of planning new ones.
- `-Agent AGENT` alone pulls that agent's entire `Models` tree plus `forge_settings.json`.
- With no arguments, the script pulls all of `Bots`.
- `-Build` overrides the selected build.
  It accepts either the display name from `builds.json` or the folder name under `Models`.
  With `-Agent` alone, it narrows the pull to that build's folder plus `builds.json`.

## Iterate after the first run

The slow parts, account setup and the first libtorch download, are done.
Each iteration has the following steps:

1. Change the training code on your machine.
1. Push the image.
   This is incremental and usually takes under 2 minutes.

   ```powershell
   ./tools/remote/push_training_image.ps1 -Tag reward-tweak-v2
   ```

   The script accepts the following options:

   - `-SkipModels` rebuilds the code only, without touching R2.
     Use it when the cloud already has the checkpoints that you want to continue.
   - `-Build BUILD` seeds that build folder during the model upload.
     It accepts a build name or folder.
     It changes only the R2 upload; the image still trains the build in its baked `forge_settings.json`.
   - `-SkipLatest` pushes only the tagged image, not `:latest`.
1. Deploy the new tag from the template, and monitor the run on W&B.
1. Stop the run and pull the results back,
   as in [Run your first remote training](#run-your-first-remote-training).

### Sync checkpoints without rebuilding

To move models without rebuilding the image, run the sync scripts directly:

```powershell
./tools/remote/push_models.ps1                                     # push what the queue needs (seed the cloud)
./tools/remote/pull_models.ps1 -Agent example-agent -TrainingOnly    # pull what the queue touched
./tools/remote/push_models.ps1 -Agent example-agent -Build MyBuild
./tools/remote/pull_models.ps1 -Agent example-agent -Build MyBuild
```

Both scripts resolve the agent through the built `ForgeConsoleTrainer`.
`push_models.ps1` and `pull_models.ps1 -Agent -TrainingOnly` transfer only what `sync-manifest` lists for the selected build's stage queue.
`pull_models.ps1 -Agent` alone pulls the agent's whole `Models` tree,
and the form without arguments pulls all of `Bots`.
To override the selected build, add `-Build BUILD`.

The manifest already includes the imported models and source segments that the build names as static opponents.
To place an imported model on R2 before you select it,
or for an agent that has no build of its own, such as a public bot,
push its whole imported-models tree:

```powershell
./tools/remote/push_models.ps1 -Agent Nexto -ImportedModels
```

> **Important:** If the cloud is in the middle of a run on a stage, don't also train that stage locally.
> The two copies can't be merged.

### Sync from the GUI

The **Remote Sync** group on the **Training** tab runs the same scripts for the selected agent and build:

- **Pull Models** pulls training files only, like `-TrainingOnly`, unless you select **All models**.
  **Include temp checkpoints** adds `-TempCheckpoints`, and **All timesteps** adds `-AllTempCheckpoints`.
- **Push Models** pushes what the queue needs.
- **Push Training Image** builds and pushes the image.
  The tag is pre-filled from `build-info`.
  `DOCKERHUB_USER` must be set before the GUI starts.

Output streams into **Console Output**, and **Cancel** stops the script and everything that it launched.
After any pull, the GUI reloads `forge_settings.json`,
so the next auto-save doesn't overwrite the segment IDs that a remote run recorded.

The GUI passes `-PlainProgress`, which every sync script accepts.
It replaces rclone's in-place `-P` display with one-line statistics every two seconds,
which suits output that goes to a log rather than a terminal.

## Back up and recover

### Periodic backup during a run

While a cloud run trains, the container exports the queue's segments, pools, and metadata to R2 on a timer,
and logs `[entrypoint] Periodic backup to R2 ...`.
A lost instance therefore costs at most one interval of progress.
The backup uses the same manifest-driven `rclone copy --update` as the final export,
plus the temporary-checkpoint copy described in the next section.

To tune the backup, set the following environment variable on the instance:

- `BACKUP_INTERVAL_SECONDS` sets the seconds between backups.
  The default is `600`, which is 10 minutes.
  `0` turns off periodic backups, so only the final export runs.

The backup does nothing when R2 isn't configured, for example during local smoke tests.

### Temporary checkpoints on R2

A segment's durable `checkpoints/` folder gains an entry when a stage stops,
through `Save(true)` on a stop or when the target is reached.
The `tsPerSave` autosaves go to the segment's `temp_checkpoints/` folder,
which the sync manifest excludes.
Without a separate copy, a crashed or pre-empted pod would lose every autosave since its last durable save.

So every export, periodic or final, also pushes the queue segments' `temp_checkpoints/` folders in a second, separate copy
(`sync-manifest --temp-checkpoints`), and logs `[entrypoint] Exporting temp checkpoints to r2:...`.

The trainer turns an autosave into a durable checkpoint in two places,
logging `Promoted autosave <timesteps> of <segment> ...`:

- **When a stage fails.**
  A stage that throws (for example, a fatal error in the learner loop) copies its newest autosave into `checkpoints/`,
  so the final export carries it like any other durable checkpoint.
- **When a segment resumes.**
  If a segment's newest autosave is newer than its newest durable checkpoint,
  the trainer copies it into `checkpoints/` before it loads.

The remote copies follow these rules:

- **A new pod resumes from the newest one.**
  Right after seeding, the entrypoint pulls the newest remote autosave of each queue segment
  and logs `[entrypoint] Seeding newest autosave ...`.
  That covers a pod that died without a final export, such as a pre-empted or killed one:
  the next launch resumes from its last pushed autosave and promotes it.
- **They're deleted at launch.**
  After that seed, the entrypoint deletes the queue segments' remote `temp_checkpoints/` folders,
  and logs `[entrypoint] Pruning the previous run's temp checkpoints ...`.
  As a result, they never grow beyond one run's worth: one checkpoint of about 25 MB per `tsPerSave`.
  The newest one survives as the new pod's promoted durable checkpoint; older ones are gone,
  so pull them first if you might want to roll back.
- **Locally, they're pulled only on request.**
  A plain `pull_models.ps1` never pulls them.
  `pull_models.ps1 -Agent AGENT -TrainingOnly -TempCheckpoints` pulls the newest one for each queue segment,
  and adding `-AllTempCheckpoints` pulls all of them.
  For more information, see [Roll back to an earlier autosave](#roll-back-to-an-earlier-autosave).
- **They upload two minutes late.**
  The copy uses `--min-age 2m`, so a checkpoint that the learner is still writing is never uploaded half-written;
  it goes up on the next backup.
- **They're non-critical.**
  A failed push or deletion logs a warning and never blocks the durable export or self-termination.

### Recover from a crash

To continue on a new pod, relaunch it.
A stage that failed has already promoted its newest autosave,
and a pod that died without exporting left its newest autosave on R2 for the next launch to seed.

To continue locally instead, pull the training files:

```powershell
./tools/remote/pull_models.ps1 -Agent AGENT -TrainingOnly
```

If the pod died without a final export, add `-TempCheckpoints` to pull its newest autosave too.
A local run promotes and resumes from that autosave automatically.

### Roll back to an earlier autosave

When training goes wrong partway through a run, you don't want the newest autosave.

1. Pull every remote autosave of each queue segment,
   which is one checkpoint of about 25 MB per `tsPerSave`:

   ```powershell
   ./tools/remote/pull_models.ps1 -Agent AGENT -TrainingOnly -TempCheckpoints -AllTempCheckpoints
   ```

1. The GUI and a local resume always use the newest checkpoint,
   so delete the timestep folders after the one that you want from the segment's local `temp_checkpoints/` folder,
   and any newer ones from its `checkpoints/` folder, including one that a failed stage promoted.
1. Optional: To continue on a new pod, copy the chosen timestep into `checkpoints/` and run `push_models.ps1`.
   Before you launch, also delete the segment's remote `temp_checkpoints/` folder,
   because the pod otherwise seeds and resumes from the newest remote autosave.

A later `-TempCheckpoints` pull downloads the newest autosave again,
so delete the unwanted folders after your last pull.

### Read the run log

The container also uploads the most recent run log, which is the trainer's full console output,
to `r2:BUCKET/latest-training.log`, on each periodic backup and on exit.
It's a single file that each run overwrites, with no per-run history,
so the console output survives a self-terminated or lost instance.
To download it, run the following command:

```powershell
rclone copyto r2:rocketforge-checkpoints/latest-training.log ./latest-training.log
```

The log upload is non-critical:
a failure logs a warning but never blocks the checkpoint export or self-termination.

### Test a running pod's model

The same pull lets you test a pod's latest model without stopping the pod:

```powershell
./tools/remote/pull_models.ps1 -Agent AGENT -TrainingOnly -TempCheckpoints
```

Then use the GUI as usual.
**Version Management** shows the segment's timesteps with an `(autosave)` suffix,
and **Export to RLBot**, the **Debug** tab's render, and the model benchmark all load the autosave.
An RLBot export of an autosave copies the inference files into the segment's `rlbot/model_snapshot/` folder,
so the export keeps working after the local `temp_checkpoints/` folder is cleaned up.

The autosave is up to one `BACKUP_INTERVAL_SECONDS` interval plus two minutes old, which is about 12 minutes by default.
For fresher models, lower `BACKUP_INTERVAL_SECONDS` on the instance.

> **Warning:** After you pull a running pod's autosave,
> don't start local training on that segment while the pod is still training it.
> The local run resumes from the autosave and becomes a second, divergent copy of the stage,
> which the newest-wins sync can't merge.

## Performance expectations

A cloud instance with a better GPU can train more slowly than your local machine, and that's expected.
Steps per second are dominated by *collection*, which is RocketSim environment stepping on the CPU, not by the GPU.
Collection is bound by per-core clock speed,
so CPU clock speed matters, not the vCPU count or the GPU tier.

Cloud hosts pair fast GPUs with many slow server cores.
For example, an AMD EPYC 7763 runs at about 1.5-2.45 GHz on all cores,
compared with about 4.5-5 GHz for a desktop,
so each cloud core collects at roughly half the rate.
In one measurement, single-thread collection was about 20,000 steps per second locally and about 10,000 on the cloud instance:
a 27-core cloud instance matched a 16-core desktop,
and the faster GPU sat idle because it wasn't the bottleneck.

To get more cloud throughput, use the following levers:

- Pick an instance with a **higher-clock CPU**.
  For this workload, clock speed matters more than core count.
- Raise **`numGames`**, the number of parallel arenas, to keep more cores busy.

To check where you stand, read the `[RLGC] Collection thread pool: ...` startup log line,
and compare with a single-thread run (`RLGC_NUM_THREADS=1`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker push` is denied. | Run `docker login -u DOCKERHUB_USER` with the access token, and confirm that the repository is `DOCKERHUB_USER/rocketforge` and private. |
| Rebuilds download libtorch again (about 250 seconds) instead of using the cache. | BuildKit's default garbage collection keeps only **20 GiB** of build cache. Past that, it evicts the least recently used records, and the libtorch layer, about 2 GB, is the largest. This is common after a fresh Docker installation, such as a reinstall or moving WSL to another drive. To raise the limit, go to **Docker Desktop > Settings > Docker Engine**, add `"builder": { "gc": { "enabled": true, "defaultKeepStorage": "50GB" } }`, and click **Apply & Restart**. To confirm, run `docker buildx du`; Reserved and Total must exceed your cache size. Alternatively, put libtorch and the other heavy dependencies in a separate base image, so that they're never rebuildable cache records. |
| No W&B run appears. | The `WANDB_API_KEY` secret is missing or mistyped in the template's environment variables. |
| `[entrypoint] R2_BUCKET not set` | The `R2_BUCKET` and `RCLONE_CONFIG_R2_*` environment variables aren't passed to the pod. For more information, see your host's guide. |
| `rclone lsf r2:rocketforge-checkpoints` fails with an authentication error locally. | Check the key and endpoint in `~/.rocketforge_env`, and make sure that the file is ASCII, with no byte order mark. If the keys are correct, the token might be scoped to a different bucket. |
| An rclone upload fails with `CreateBucket ... 403 AccessDenied`. | The bucket-scoped R2 token can't create buckets, and rclone tries to before uploading. Set `RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true`, which is in the environment file template and baked into the container. Rebuild and redeploy the image. For a quick local check, prefix the command: `RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true rclone copyto ...`. |
| The log shows an export `WARNING` or `Export FAILED — idling`. | An `rclone copy` call failed, and the checkpoints are still on the pod's disk. Fix the credentials, run the export again, and only then terminate the pod. Never terminate first. |
| The host has a deployment or self-termination problem. | See your host's guide: [Vast.ai](VastAI.md) or [RunPod](RunPod.md). |
| Steps per second are lower than locally. | Collection is usually CPU-bound, not GPU-bound, and slower server cores make this expected. For more information, see [Performance expectations](#performance-expectations). The pool caps itself to the cgroup CPU quota and pins to one NUMA node; check the `[RLGC] Collection thread pool: ...` startup log line. |
| The run unexpectedly starts from scratch. | The selected build's checkpoints weren't in R2, so run `push_models.ps1` first. Alternatively, the sync manifest couldn't be computed, and the log shows `[entrypoint] WARNING: could not compute the sync manifest — skipping seed.`; check the agent name and the `buildName` in `forge_settings.json`. |
| `Imported model folder not found: <agent> / ...` at startup | A selected static opponent's model files never reached the pod. Check the `[entrypoint] Seeding training manifest ...` output in the run log, and push the files from your machine with `push_models.ps1`, or with `push_models.ps1 -Agent NAME -ImportedModels`. |
