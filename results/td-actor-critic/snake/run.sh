python3 NNtesting/experiments/td_actor_critic/snake_td.py --out original.jsonl --arm original --seeds 11-20
python3 NNtesting/experiments/td_actor_critic/snake_td.py --out td.jsonl --arm td --seeds 11-20 --set post=advantage --set critic_rate=0.03 --set actor_rate=0.1 --set tau=0.03
python3 NNtesting/experiments/td_actor_critic/snake_td.py --out reward.jsonl --arm reward --seeds 11-20 --set post=advantage --set actor_rate=0.03 --set tau=0.03
python3 NNtesting/experiments/td_actor_critic/snake_td.py --out td_r1.jsonl --arm td --seeds 11-20 --set post=advantage --set critic_rate=0.03 --set actor_rate=0.03 --set tau=0.03
