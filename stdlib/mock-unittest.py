import unittest
from unittest import mock


class TestMethods(unittest.TestCase):
    @mock.patch("sklearn.preprocessing.StandardScaler")
    @mock.patch("sklearn.preprocessing.StandardScaler")
    def test_correct(self, c1, c2):
        import sklearn

        print(c1)
        print(c2)
        assert c1 is sklearn.preprocessing.StandardScaler


if __name__ == "__main__":
    unittest.main()
