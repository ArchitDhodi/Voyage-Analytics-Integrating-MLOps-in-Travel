"""
Run all trainers: price, gender, recommender.
"""
from train_price import main as train_price
from train_gender import main as train_gender
from train_recommender import main as train_recommender


def main():
    print("== Training price model ==")
    train_price()
    print("== Training gender model ==")
    train_gender()
    print("== Training recommender ==")
    train_recommender()
    print("All training steps completed.")


if __name__ == "__main__":
    main()
