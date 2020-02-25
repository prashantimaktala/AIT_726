from xml.etree import ElementTree
import pandas as pd


def load_dataset(path):
    tree = ElementTree.parse(path)
    root = tree.getroot()
    reviews = []
    for review in root:
        for sentence in review.findall("./sentences/sentence"):
            opinions = sentence.findall("./Opinions/Opinion")
            for opinion in opinions:
                record = {}
                record.update(sentence.attrib)
                record.update({'text': sentence.find('text').text})
                record.update(opinion.attrib)
                reviews.append(record)
    return pd.DataFrame(reviews)
