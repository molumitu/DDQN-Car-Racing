from ray.rllib.agents.dqn import DQNTrainer
import ray.tune
from GameEnv import RacingEnv
config = {
    "env": RacingEnv,
    "seed":73,
    "framework":"tf2",
    "prioritized_replay": True,
    "dueling": ray.tune.grid_search([False, True]),
    "double_q": ray.tune.grid_search([False, True]),
}

trainer = DQNTrainer(config)
# while True:
#     trainer.train()
ray.tune.run(
    DQNTrainer,
    config=config,
    stop={
        "episode_reward_mean": 50,
        "agent_timesteps_total": 200000,
    },
    checkpoint_freq=10,
    checkpoint_at_end=True,
    verbose=1,
    progress_reporter=ray.tune.CLIReporter(print_intermediate_tables=True)
)
