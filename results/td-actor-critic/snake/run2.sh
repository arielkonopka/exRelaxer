python3 NNtesting/experiments/td_actor_critic/snake_td.py --out reward_r3.jsonl --arm reward --seeds 11-20 --set post=advantage --set actor_rate=0.1 --set tau=0.03
python3 NNtesting/experiments/td_actor_critic/snake_td.py --out reward_r03.jsonl --arm reward --seeds 11-20 --set post=advantage --set actor_rate=0.01 --set tau=0.03
python3 NNtesting/experiments/td_actor_critic/snake_td.py --out td_r03.jsonl --arm td --seeds 11-20 --set post=advantage --set critic_rate=0.03 --set actor_rate=0.01 --set tau=0.03
python3 NNtesting/experiments/td_actor_critic/snake_td.py --out td_taken.jsonl --arm td --seeds 11-20 --set post=taken --set critic_rate=0.03 --set actor_rate=0.03 --set tau=0.03
