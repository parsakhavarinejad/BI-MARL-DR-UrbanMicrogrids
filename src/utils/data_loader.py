import pandas as pd
import numpy as np


class SmartGridDataLoader:
    """
    Load and preprocess smart-grid time series data, then pack it into daily episodes.

    The main goal is to convert the raw panel data into a 3D tensor per day with shape:
        (num_homes, episode_len, num_features)

    In the common 15-minute resolution setting:
        episode_len = 96  (96 * 15min = 24h)

    Example (first experiment):
        (12, 96, 8)  -> 12 homes, 96 time steps, 8 features

    Parameters
    ----------
    data_path : str
        Path to the dataset file in CSV.GZ format.

    Attributes
    ----------
    df : pandas.DataFrame
        Loaded and preprocessed data.
    feature_cols : list[str]
        Feature columns used to build the episode tensor.
    home_uinque : list
        Sorted unique home IDs found in the dataset.
    date_unique : list
        Sorted unique dates found in the dataset.
    daily_episodes : np.ndarray
        Array of episodes with shape (num_days, num_homes, episode_len, num_features).
    valid_date : list[str]
        Dates that were successfully converted into complete episodes.

    Methods
    -------
    get_episode(index)
        Return the episode tensor for a given day index.
    __len__()
        Return number of valid daily episodes.
    """

    def __init__(self, data_path):
        """Initialize the loader, build features, and prepare daily episodes."""
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
            self.df["ts"].dt.hour * 60 + self.df["ts"].dt.minute
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
        """
        Convert the dataframe into a list of daily episode tensors.

        A day is considered valid only if it contains a complete grid of:
            episode_len time steps for each home.

        Notes
        -----
        This method fills:
            - self.daily_episodes
            - self.valid_date
        """
        episode_len = 96  # 96 slices of 15 minutes in a day
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
        """
        Get a daily episode tensor by index.

        Parameters
        ----------
        index : int
            Day index in the prepared episode list.

        Returns
        -------
        np.ndarray
            Episode tensor with shape (homes, steps, features).
        """
        return self.daily_episodes[index]

    def __len__(self):
        """Return the number of valid daily episodes."""
        return len(self.daily_episodes)


# -------- Test the script --------
if __name__ == "__main__":
    data_loader = SmartGridDataLoader(r"data\\IDEAL\\panel_env_ready_15m.csv.gz")
    episode = data_loader.get_episode(index=1)
    print(episode.shape)
