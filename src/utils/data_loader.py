import pandas as pd
import numpy as np

class SmartGridDataLoader:
    def __init__(self, data_path, num_agents=25):
        print(f"Loading the data from {data_path}")
        self.df = pd.read_csv(data_path, compression="gzip")
        self.target_agents = num_agents

        print(f"Selecting the top {num_agents} agents with the most data...")
        
        home_counts = self.df.groupby("homeid")["date"].nunique()
        best_homes = home_counts.sort_values(ascending=False).head(num_agents).index.tolist()
        
        self.df = self.df[self.df["homeid"].isin(best_homes)].copy()
        print(f"Kept agents: {best_homes}")

        print("Calculating common dates for these best agents...")
        home_dates = self.df.groupby("homeid")["date"].unique()
        
        if len(home_dates) > 0:
            common_dates = set(home_dates.iloc[0])
            for dates in home_dates.iloc[1:]:
                common_dates = common_dates.intersection(dates)
            
            common_dates = sorted(list(common_dates))
            print(f"Found {len(common_dates)} days where ALL {num_agents} best agents are present.")
            
            self.df = self.df[self.df["date"].isin(common_dates)].copy()
        else:
            common_dates = []

        self.df["ts"] = pd.to_datetime(self.df["ts"])
        self.df["date"] = self.df["ts"].dt.date.astype("str")
        self.df = self.df.sort_values(by=["date", "homeid", "ts"])
        
        minutes_in_day = 24 * 60
        time_fraction = (self.df["ts"].dt.hour * 60 + self.df["ts"].dt.minute) / minutes_in_day
        self.df["tod_sin"] = np.sin(time_fraction * 2 * np.pi)
        self.df["tod_cos"] = np.cos(time_fraction * 2 * np.pi)
        
        day_fraction = (pd.to_datetime(self.df["date"]).dt.dayofweek) / 7
        self.df["day_of_week_sin"] = np.sin(day_fraction * 2 * np.pi)
        self.df["day_of_week_cos"] = np.cos(day_fraction * 2 * np.pi)
        
        self.home_unique = sorted(self.df["homeid"].unique())
        
        self.price_min = self.df["unit_charge_pence_per_kwh"].min()
        self.price_max = self.df["unit_charge_pence_per_kwh"].max()
        self.df["price_norm"] = (self.df["unit_charge_pence_per_kwh"] - self.price_min) / (self.price_max - self.price_min + 1e-6)

        self.load_clip = self.df["e_kwh"].quantile(0.99)
        self.df["e_kwh_norm"] = np.clip(self.df["e_kwh"], 0, self.load_clip) / (self.load_clip + 1e-6)
        
        self.df["e_kwh_lag1"] = self.df.groupby("homeid")["e_kwh_norm"].shift(1, fill_value=0)
        self.df["e_kwh_roll_mean_4"] = self.df.groupby("homeid")["e_kwh_norm"].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )
        
        self.tmp_mean = self.df["temperature"].mean()
        self.tmp_std = self.df["temperature"].std()
        self.df["temp_norm"] = (self.df["temperature"] - self.tmp_mean) / (self.tmp_std + 1e-6)

        self.df["income_norm"] = self.df["income_band"] / (self.df["income_band"].max() + 1e-6)
        if self.df["urban_rural_class"].dtype == 'object':
             self.df["urban_rural_class"] = self.df["urban_rural_class"].replace('3+', 3).astype('int')
        self.df["urban_norm"] = self.df["urban_rural_class"] / (self.df["urban_rural_class"].max() + 1e-6)

        self.df = self.df.sort_values(by=["date", "homeid", "ts"])
        self.date_unique = sorted(self.df["date"].unique())

        print(f"There are {len(self.home_unique)} homes in total and {len(self.date_unique)} days!")

        self.feature_cols = [
            "e_kwh_norm",       
            "price_norm",      
            "e_kwh_lag1",       
            "e_kwh_roll_mean_4",
            "tod_sin",          
            "tod_cos",
            "day_of_week_sin", 
            "day_of_week_cos",
            "temp_norm",        
            "income_norm",     
            "urban_norm"        
        ]

        self.state_dim = len(self.feature_cols)
        self.daily_episodes = []
        self._process_episodes()

    def _process_episodes(self):
        episode_len = 96
        valid_episode = []
        
        print(f"Processing {len(self.date_unique)} episodes...")
        
        gb_date = self.df.groupby("date")
        
        for date in self.date_unique:
            if date not in gb_date.groups: continue
            
            df_day = gb_date.get_group(date)
            homes_data = []
            
            for home_id, df_agent in df_day.groupby("homeid"):
                if len(df_agent) == episode_len:
                    homes_data.append(df_agent[self.feature_cols].values)
            
            if len(homes_data) == self.target_agents:
                day_tensor = np.stack(homes_data)
                valid_episode.append(day_tensor)

        self.daily_episodes = np.array(valid_episode)
        print(f"Final Dataset: {len(self.daily_episodes)} clean episodes.")
        print(f"Tensor Shape: {self.daily_episodes.shape}")

    def get_episode(self, index):
        return self.daily_episodes[index]

    def __len__(self):
        return len(self.daily_episodes)

if __name__ == "__main__":
    data_loader = SmartGridDataLoader("data/IDEAL/panel_env_ready_15m_50agents.csv.gz", num_agents=25)
    if len(data_loader) > 0:
        episode = data_loader.get_episode(index=0)
        print(episode.shape)