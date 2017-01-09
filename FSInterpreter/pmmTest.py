from phraseModelMgr import *


def main():
    pmm = PhraseModelMgr()

    test_phrases = [("HI MYNA ME IS ROB BOYER", 1.0),
                    ("HI MY NAME IS ROB BOYER", 0.66),
                    ("HI MY SAME IS ROB BOYER", 0.33),
                    ("I MY NAME IS ROB BOYER", 0.1),
                    ("I MYNA ME IS ROB BOYER", 0.05)]

    print('Best when empty: %s, Conf=%.2f' % pmm.get_best_phrase())

    for (phrase, conf) in test_phrases:
        print(phrase, pmm.add_phrase(phrase, conf))
        print('Best: %s, Conf=%.2f' % pmm.get_best_phrase())

    pmm.dump_phrases()
    pmm.reset()

    for (phrase, conf) in test_phrases:
        print(phrase, pmm.add_phrase(phrase, conf))
        print('Best: %s, Conf=%.2f' % pmm.get_best_phrase())

    pmm.dump_phrases()
    pmm.reset()


if __name__ == '__main__':
    main()
