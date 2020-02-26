import unittest

from absa import model


class EvaluateModelTestCase(unittest.TestCase):
    def test_evaluate_model(self):
        score = model.evaluate([1], [1])
        print('\n%s' % score.to_string())
        self.assertIsNotNone(score)

    def test_score_index(self):
        score = model.evaluate([1], [1])
        print('\nAccuracy: %0.2f' % score.accuracy)
        self.assertIsNotNone(score.accuracy)


if __name__ == '__main__':
    unittest.main()
