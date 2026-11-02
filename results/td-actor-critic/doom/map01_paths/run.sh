# Run from a folder holding the doom_td run directories (each with config.json + checkpoint.pkl).
python3 NNtesting/experiments/td_actor_critic/where_agents_go.py --runs exit/approach_s1 exit/approach_s2 exit/approach_s3 --games 4 --out approach.png
python3 NNtesting/experiments/td_actor_critic/where_agents_go.py --runs killer/orient_s1 killer/orient_s2 killer/orient_s3 --games 4 --out orient.png
python3 NNtesting/experiments/td_actor_critic/where_agents_go.py --runs killer/orient_s1 killer/orient_s2 killer/orient_s3 freeze/freeze_s1 freeze/freeze_s2 freeze/freeze_s3 --games 4 --seed 2 --out recorded_paths.png --record videos
