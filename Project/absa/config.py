from os.path import join as join_path
from os.path import dirname as dirname
from os.path import abspath as abspath

from absa.utils import Object

PROJECT_PATH = join_path(dirname(abspath(__file__)), '..')

DATA_PATH = {
    'asba.foursquare.raw.test.a': join_path(PROJECT_PATH, 'data', 'raw', 'foursquare_asba', 'foursquare_testA.xml'),
    'asba.foursquare.raw.test.b': join_path(PROJECT_PATH, 'data', 'raw', 'foursquare_asba', 'foursquare_testB.xml'),
    'asba.foursquare.raw.test.gold': join_path(PROJECT_PATH, 'data', 'raw', 'foursquare_asba', 'foursquare_gold.xml'),
    'asba.semeval16.raw.train': join_path(PROJECT_PATH, 'data', 'raw', 'semeval16_asba',
                                          'ABSA16_Restaurants_Train_SB1_v2.xml'),
    'asba.semeval16.raw.test.a': join_path(PROJECT_PATH, 'data', 'raw', 'semeval16_asba', 'EN_REST_SB1_TEST.A.xml'),
    'asba.semeval16.raw.test.b': join_path(PROJECT_PATH, 'data', 'raw', 'semeval16_asba', 'EN_REST_SB1_TEST.B.xml'),
    'asba.semeval16.raw.test.gold': join_path(PROJECT_PATH, 'data', 'raw', 'semeval16_asba',
                                              'EN_REST_SB1_TEST.gold.xml'),
}

datasets = Object()

datasets.foursquare = Object()
datasets.foursquare.raw = Object()
datasets.foursquare.raw.test = Object()
datasets.foursquare.raw.test.a = 'datasets.foursquare.raw.test.a'
datasets.foursquare.raw.test.b = 'datasets.foursquare.raw.test.b'
datasets.foursquare.raw.test.gold = 'datasets.foursquare.raw.test.gold'

datasets.semeval16 = Object()
datasets.semeval16.raw = Object()
datasets.semeval16.raw.train = 'asba.semeval16.raw.train'
datasets.semeval16.raw.test = Object()
datasets.semeval16.raw.test.a = 'asba.semeval16.raw.test.a'
datasets.semeval16.raw.test.b = 'asba.semeval16.raw.test.b'
datasets.semeval16.raw.test.gold = 'asba.semeval16.raw.test.gold'
