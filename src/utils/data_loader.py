import pandas as pd
import numpy as np


class SmartGridDataLaoder:
    """
    In this class we preprocess the data and turn it to different episodes,
    we reshape the data to (num_homes, length of the episode, the number of features)

    In our first experiment the reshaped data has (12, 96, 3)

    class SmartGridDataLoader
        input (str): the data path - in csv.gz format
        output (np.array): daily episode per index
        core method: get_episode
    """

    def __init__(self, data_path):
        print(f"Loading the data from {data_path}")
        self.df = pd.read_csv(data_path, compression="gzip")

        self.df["ts"] = pd.to_datetime(self.df["ts"])
        self.df["date"] = self.df["ts"].dt.date.astype("str")
        self.df["e_kwh_lag1"] = self.df.groupby("homeid")["e_kwh"].shift(
            1, fill_value=0
        )
        self.df["e_kwh_roll_mean_4"] = self.df.groupby("homeid")["e_kwh"].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )

        minutes_in_day = 24 * 60
        time_fraction = (
            pd.to_datetime(self.df["ts"]).dt.hour * 60
            + pd.to_datetime(self.df["ts"]).dt.minute
        ) / minutes_in_day

        self.df["tod_sin"] = np.sin(time_fraction * 2 * np.pi)
        self.df["tod_cos"] = np.cos(time_fraction * 2 * np.pi)

        day_fraction = (pd.to_datetime(self.df["date"]).dt.dayofweek) / 7
        self.df["day_of_week_sin"] = np.sin(day_fraction * 2 * np.pi)
        self.df["day_of_week_cos"] = np.cos(day_fraction * 2 * np.pi)

        self.df = self.df.sort_values(by=["date", "homeid", "ts"])
        self.home_uinque = sorted(self.df["homeid"].unique())
        self.date_unique = sorted(self.df["date"].unique())

        print(
            f"There are {len(self.home_uinque)} homes in total and {len(self.date_unique)} days!"
        )

        self.daily_episodes = []
        self.valid_date = []

        self.feature_cols = [
            "e_kwh",
            "unit_charge_pence_per_kwh",
            "e_kwh_lag1",
            "e_kwh_roll_mean_4",
            "tod_sin",
            "tod_cos",
            "day_of_week_sin",
            "day_of_week_cos",
        ]

        self._process_episodes()

    def _process_episodes(self):

        episode_len = 96  # we have 96 numbers of 15min slice in a day
        num_homes = len(self.home_uinque)
        expected_rows = episode_len * num_homes
        valid_episode = []

        for date in self.date_unique:

            df_dates = self.df[self.df["date"] == date]
            if len(df_dates) != expected_rows:
                print(f"Skipping incomplete date for {date}")
                continue

            values = df_dates[self.feature_cols].values

            try:
                episode_matrix = values.reshape(
                    num_homes, episode_len, len(self.feature_cols)
                )
                valid_episode.append(episode_matrix)
                self.valid_date.append(date)
            except ValueError as e:
                print(f"Processing failed for {date}: {e}")

        self.daily_episodes = np.array(valid_episode)
        print(f"Successfully prepared {len(self.daily_episodes)} community episodes.")

    def get_episode(self, index):
        """Returns (homes, steps, features) for a specific day index."""
        return self.daily_episodes[index]

    def __len__(self):
        return len(self.daily_episodes)


if __name__ == "__main__":
    data_loader = SmartGridDataLaoder("data\IDEAL\panel_env_ready_15m.csv.gz")
    episode = data_loader.get_episode(index=1)
    print(episode.shape)
