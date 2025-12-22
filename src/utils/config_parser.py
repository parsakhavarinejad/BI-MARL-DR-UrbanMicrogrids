import yaml


class Config:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.actor_state_dim = int(cfg["networks"]["actor"]["state_dim"])
        self.actor_action_dim = int(cfg["networks"]["actor"]["action_dim"])
        self.actor_lr = float(cfg["networks"]["actor"]["lr"])

        self.critic_global_state_dim = int(
            cfg["networks"]["critic"]["global_state_dim"]
        )
        self.critic_lr = float(cfg["networks"]["critic"]["lr"])

        self.k_epochs = int(cfg["agent"]["k_epochs"])
        self.gamma = float(cfg["agent"]["gamma"])
        self.eps_clip = float(cfg["agent"]["eps_clip"])

        self.num_episodes = int(cfg["main"]["num_episodes"])
        self.num_agents = int(cfg["main"]["num_agents"])
        self.update_interval = int(cfg["main"]["update_interval"])
        self.num_steps_per_day = int(cfg["main"]["num_steps_per_day"])

        self.data_path = cfg["data"]["path"]

        self.agent_entropy_coeff = float(cfg["agent"]["entropy_coeff"])

        self.env_ratio_clip_min_max = list(cfg["env"]["ratio_clip_min_max"])
        self.env_total_price_clip_min_max = list(cfg["env"]["total_price_clip_min_max"])
        self.env_scaling_factor = float(cfg["env"]["scaling_factor"])
        self.env_discomfort_weight = float(cfg["env"]["discomfort_weight"])
