#Tests for data_parsing.ipynb


import pandas as pd
from data_parsing import DataParsing
import unittest

class TestDataParsing(unittest.TestCase): 

    def setUp(self):
        self.dp = DataParsing(pd.read_parquet("data/Gridmode-00000-of-00001.parquet"))

    def test_get_attributes(self):
        expected = pd.Series(['House', 'Name', 'Nationality', 'BookGenre', 'Food', 'Color', 'Animal'])
        result = self.dp.get_csp()[0]['variables']
        self.assertTrue(result.equals(expected))

    def test_get_domains(self):
        expected = pd.Series([''])
    

if __name__ == '__main__':
    unittest.main()