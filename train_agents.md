We provide scripts to train and test PQN agents on the environments.

First, make sure that any custom environments and mods are registered in `src/jaxatari/core.py`.

## Training a PQN agent
1. Adapt the corresponding config script located in `scripts/benchmarks/config/alg/` to use your environment and modification. 
    - Note the difference between the pixel based and object-centric config.

2. Run the config via `bash python pqn_agent.py +alg=pqn_jaxatari_<mode>`
    - During training, you can monitor the training curves with WandB. 
    - The most important value is: `returned_episode_returns` under `Charts`. If this is not increasing over time, something is probably off.
    - For debugging, enable `TEST_DURING_TRAIN` and `RECORD_VIDEO`. This will log videos under `Media` to WandB and you can watch what the agent does.
    - Once done training, compile time and run time will be reported.

## Evaluating a PQN agent
1. Adapt your previous config and make sure to set `MOD_NAME` to your corresponding mod (if applicable), as well as `RECORD_VIDEO=True` if you want to log videos of the test environments.
2. Run the config via `bash python pqn_test.py +alg=pqn_jaxatari_<mode>`
    - Once done, average return and length will be reported. 
