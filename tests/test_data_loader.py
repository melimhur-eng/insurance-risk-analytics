from src.data_loader import load_data


def test_load_data():

    df = load_data(
        "data/MachineLearningRating_v3.txt"
    )

    assert df is not None
    assert len(df) > 0