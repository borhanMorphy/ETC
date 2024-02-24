import random
from datetime import datetime, timedelta


import torch
from torch.utils.data import IterableDataset

class DummyDataset(IterableDataset):
    def __init__(self, num_samples: int, callback=None):
        self._num_samples = num_samples
        self._callback = callback

    @staticmethod
    def generate_random_time():
        start_date = datetime(2018, 1, 1)  # Start date
        end_date = datetime.now()  # Today's date
        random_time = start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))
        return random_time

    @staticmethod
    def random_date_formats():
        date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%Y", "%Y.%m.%d", "%d %B %Y", "%B %d, %Y", "%d %B %Y", "%Y %B %d"]

        # Randomly select two different formats
        format1 = random.choice(date_formats)
        format2 = random.choice([fmt for fmt in date_formats if fmt != format1])

        return format1, format2

    def generate(self):
        for _ in range(self._num_samples):
            same = random.random() > 0.5

            format_1, format_2 = self.random_date_formats()

            if same:
                dt = self.generate_random_time()
                # Format the datetime with selected formats
                formated_dt_1 = dt.strftime(format_1)
                formated_dt_2 = dt.strftime(format_2)
            else:
                dt_1 = self.generate_random_time()
                dt_2 = self.generate_random_time()

                # Format the datetime with selected formats
                formated_dt_1 = dt_1.strftime(format_1)
                formated_dt_2 = dt_2.strftime(format_2)

            label = 1 if same else 0
            data = f"{formated_dt_1}|{formated_dt_2}"

            if self._callback:
                yield {
                    "token_ids": torch.LongTensor(self._callback(data)),
                    "label": torch.FloatTensor([label]),
                    "date_1": formated_dt_1,
                    "date_2": formated_dt_2,
                }
            else:
                yield data, label


    def __iter__(self):
        return iter(self.generate())

    def __len__(self) -> int:
        return self._num_samples

