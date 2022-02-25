from GameEnv import RacingEnv
import ray.rllib.agents.dqn as dqn
PATH = R"C:/Users/zgj_t/ray_results/DQNTrainer_2022-02-25_16-32-16/DQNTrainer_RacingEnv_70340_00000_0_double_q=False,dueling=False_2022-02-25_16-32-16/checkpoint_000085/checkpoint-85"
PATH = R"C:/Users/zgj_t/ray_results/DQNTrainer_2022-02-25_18-32-16/DQNTrainer_RacingEnv_33cb1_00003_3_double_q=True,dueling=True_2022-02-25_18-32-16/checkpoint_000130/checkpoint-130"
# PATH = R"C:/Users/zgj_t/ray_results/DQNTrainer_2022-02-25_18-32-16/DQNTrainer_RacingEnv_33cb1_00003_3_double_q=True,dueling=True_2022-02-25_18-32-16/checkpoint_000140/checkpoint-140"
# PATH = R"C:/Users/zgj_t/ray_results/DQNTrainer_2022-02-25_18-32-16/DQNTrainer_RacingEnv_33cb1_00000_0_double_q=False,dueling=False_2022-02-25_18-32-16/checkpoint_000120/checkpoint-120"
config = {
    "env": RacingEnv,
    "seed":20,
    "framework":"tf2",
    "prioritized_replay": True,
    "dueling": True,
    "double_q": True,
}

trainer = dqn.DQNTrainer(config)
trainer.restore(PATH)

episode_reward = 0
done = False
env = RacingEnv()
obs = env.reset()
while not done:
    action = trainer.compute_single_action(obs)
    obs, reward, done, info = env.step(action)
    env.render(action)
    episode_reward += reward
print(episode_reward)
