import unittest

from absa.config import DATA_PATH
from absa.dataset import load_dataset


class LoadDatasetTestCase(unittest.TestCase):
    def test_load_dataset(self):
        df = None
        for keys in DATA_PATH.keys():
            df = load_dataset(DATA_PATH[keys])
            if df is None:
                break
        self.assertIsNotNone(df)


if __name__ == '__main__':
    unittest.main()
