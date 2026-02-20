# Hydra + Optuna Pruning — a Minimal Experiment Pattern

> *Pruning* as in **Optuna trial pruning** (stopping unpromising hyperparameter trials early to save time and compute), not model pruning or weight sparsification.

This repository is a small, self-contained experiment (not a  package) that grew out of a practical question:

> *How do you keep Hydra for clean experiment configuration, while still using Optuna’s pruning features during training?*

If pruning is not required, Hydra’s built-in Optuna sweeper is often simpler and perfectly sufficient.



---

## The problem that motivated this

Optuna supports *trial-level early stopping*: during training, a trial can report intermediate metrics and decide to stop itself early if it looks unpromising.

To do that, the training loop needs access to an `optuna.trial.Trial` object so it can:
- report intermediate values (e.g. validation accuracy per epoch),
- check whether the trial should be pruned,
- and exit early when appropriate.

Hydra has an official Optuna integration that works very well for defining search spaces and launching sweeps. However, that integration treats each trial as a black-box run and does not naturally expose the `Trial` object inside the training loop.

This creates a tension:
- Hydra excels at configuration, composition, and experiment hygiene.
- Optuna excels at optimization logic and pruning.
- Trying to make Hydra “own” the optimization loop makes pruning awkward.

---

## The idea

Hydra is used here as an **experiment runtime**:
- it loads and composes configuration,
- allows command-line overrides, 
- and gives each run a clean, reproducible environment.

Optuna is used **explicitly** for what it is best at:
- defining and managing the study,
- enqueueing known baseline trials,
- suggesting hyperparameters,
- and pruning trials during training.

Rather than asking Hydra to manage the optimization loop, the loop is written directly with Optuna, and Hydra simply provides the configuration that flows through it.



The hyperparameter search space lives in YAML, close to the rest of the experiment configuration.  
A thin “suggestion” layer converts that declarative specification into Optuna suggestions and applies them back into the Hydra config for each trial.    
The training loop reports validation accuracy at each epoch and lets Optuna decide whether the trial should continue.  



Search ranges and choices can be edited directly in configs/search.yaml or in command line:  
```bash
python src/main.py train.epochs=20 data.batch_size=256
```


---



## Demo

To keep things concrete, i used a very small MNIST setup:
- a simple MLP or CNN,
- a short training loop,
- and a validation metric reported every epoch.

Some trials run to completion. Others are stopped early when they clearly underperform.  
This makes pruning visible and easy to reason about without a large codebase.

---

## Running it

```bash
pip install -r requirements.txt
python src/main.py






