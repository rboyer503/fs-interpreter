from wmmWrapper import *
from time import sleep

#           1111111111222222
# 01234567890123456789012345
# ABCDEFGHIJKLMNOPQRSTUVWXYZ


def main():
    wmm = WordModelMgr()
    if not wmm.initialize():
        print("Initialize failed.")
        return

    wmm.add_letter_prediction(2, 0.992046116107, 0.1)  # C
    sleep(0.35)
    wmm.add_letter_prediction(14, 0.993403973465, 0.1)  # O
    sleep(0.05)
    wmm.add_letter_prediction(1, 0.421154717222, 0.3)
    sleep(0.005)
    wmm.add_letter_prediction(2, 0.200448605746, 0.3)
    sleep(0.2)
    wmm.add_letter_prediction(20, 0.106096796237, 0.75)
    sleep(0.1)
    wmm.add_letter_prediction(12, 0.993403965129, 0.9)  # M
    sleep(0.45)
    wmm.add_letter_prediction(14, 0.993403928753, 0.1)  # O
    sleep(0.08)
    wmm.add_letter_prediction(6, 0.611820873989, 0.1)
    sleep(0.08)
    wmm.add_letter_prediction(23, 0.20870134994, 0.1)
    sleep(0.23)
    wmm.add_letter_prediction(13, 0.993112756251, 0.1)  # N
    sleep(0.03)
    wmm.add_letter_prediction(19, 0.172608551531, 0.1)
    sleep(0.2)
    wmm.add_letter_prediction(11, 0.236728259535, 0.1)
    sleep(0.08)
    wmm.add_letter_prediction(2, 0.165517257023, 0.1)

    wmm.finalize_prediction()
    wmm.dump_candidates()

    print("BEST:", wmm.get_best_prediction().decode())


if __name__ == '__main__':
    main()
