import unittest

from absa.config import DATA_PATHS
from absa.dataset import load_dataset


class LoadDatasetTestCase(unittest.TestCase):
    def test_load_dataset(self):
        df = None
        for keys in DATA_PATHS.keys():
            df = load_dataset(DATA_PATHS[keys])
            print(keys, df.shape)
        self.assertIsNotNone(df)


if __name__ == '__main__':
    unittest.main()
