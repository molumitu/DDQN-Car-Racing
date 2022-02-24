from ray.rllib.agents.dqn import DQNTrainer
import ray.tune
from GameEnv import RacingEnv
config = {
    "env": RacingEnv,
    "seed":73,
    "framework":"torch",

    "prioritized_replay": ray.tune.grid_search([False, True]),
    "dueling": ray.tune.grid_search([False, True]),
    "double_q": ray.tune.grid_search([False, True]),
    # "num_atoms": 0,
}

trainer = DQNTrainer(config)
ray.tune.run(
    DQNTrainer,
    config=config,
    stop={
        "episode_reward_mean": 300,
        "agent_timesteps_total": 200000,
    },
    checkpoint_at_end=True,
    verbose=1,
    progress_reporter=ray.tune.CLIReporter(print_intermediate_tables=True)
)
