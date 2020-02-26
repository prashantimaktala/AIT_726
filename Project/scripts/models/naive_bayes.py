from absa import model


def main():
    scores = model.evaluate([1], [1])
    print(scores.to_string())


if __name__ == '__main__':
    main()
