import _thread
import site
import sys
import json
import os

wandb_run = None
stop_requested = False

def _record_stop_request(*args, **kwargs):
	global stop_requested
	if not stop_requested:
		print("wandb: stop requested from the website.")
	stop_requested = True

# wandb reacts to the website's Stop button by faking a Ctrl+C in the main thread
# (_thread.interrupt_main() in older SDKs, wandb.sdk.lib.interrupt.interrupt_main() in newer ones).
# In this embedded interpreter that either never fires or surfaces as a KeyboardInterrupt inside add_metrics,
# so record the request instead and let the trainer poll it through add_metrics' return value.
def _intercept_stop_signal():
	_thread.interrupt_main = _record_stop_request
	try:
		from wandb.sdk.lib import interrupt
		interrupt.interrupt_main = _record_stop_request
	except ImportError:
		pass

# Takes in the python executable path, the three wandb init strings, and optionally the current run ID
# Returns the ID of the run (either newly created or resumed)
def init(py_exec_path, project, group, name, id = None):

	global wandb_run, stop_requested
	stop_requested = False

	# Fix the path of our interpreter so wandb doesn't run RLGym_PPO instead of Python
	# Very strange fix for a very strange problem
	sys.executable = py_exec_path

	try:
		site_packages_dir = os.path.join(os.path.join(os.path.dirname(py_exec_path), "Lib"), "site-packages")
		sys.path.append(site_packages_dir)
		site.addsitedir(site_packages_dir)
		import wandb
	except Exception as e:
		raise Exception(f"""
			FAILED to import wandb! Make sure GigaLearnCPP isn't using the wrong Python installation.
			This installation's site packages: {site.getsitepackages()}
			Exception: {repr(e)}"""
		)

	_intercept_stop_signal()

	print("Calling wandb.init()...")
	if not (id is None) and len(id) > 0:
		wandb_run = wandb.init(project = project, group = group, name = name, id = id, resume = "allow")
	else:
		wandb_run = wandb.init(project = project, group = group, name = name)
	return wandb_run.id

# Returns True once the run has been stopped from the wandb website.
def add_metrics(metrics):
	global wandb_run
	try:
		wandb_run.log(metrics)
	except KeyboardInterrupt:
		# Fallback for an SDK whose stop signal slipped past _intercept_stop_signal.
		_record_stop_request()
	return stop_requested

def finish():
	global wandb_run
	if wandb_run is not None:
		wandb_run.finish()
		wandb_run = None
