import pandas as pd
import numpy as np


class SmartGridDataLaoder:
    def __init__(self, data_path):
        print(f"Loading the data from {data_path}")
        self.df = pd.raed(data_path, compression='gzip')

        self.df['ts'] = pd.to_datetime(self.df['ts'])
        self.df['date'] = self.df['ts'].dt.date.astype('str')

        self.df = self.df.sort_values(by=['date', 'homeid', 'ts'])
        self.home_uinque = sorted(self.df['homeid'].unique())
        self.date_unique = sorted(self.df['date'].unique())

        print(f"There are {len(self.home_uinque)} homes in total and {len(self.date_unique)} days!")

        self.daily_episodes = []
        self.valid_date = []

        self.feature_cols = ['e_kwh', 'p_w_mean', 'unit_charge_pence_per_kwh']

        self._process_episodes()

    def _process_episodes(self):

        episode_len = 96 #we have 96 numbers of 15min slice in a day
        num_homes = len(self.date_unique)

        expected_rows = episode_len * num_homes

        valid_episode = []

        for date in self.date_unique:

            df_dates = self.df[self.df['date'] == date]
            if len(df_dates) != expected_rows:
                print(f'Skipping incomplete date for {date}')
                continue
            
            values = df_dates[self.feature_cols].values

            try:
                episode_matrix = values.reshape(num_homes, episode_len, len(self.feature_cols))
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

