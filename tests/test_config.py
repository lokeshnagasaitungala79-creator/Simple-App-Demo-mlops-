import pytest

from prediction_service.prediction import form_response, api_response
import prediction_service


input_data = {
    "Incorrect Range": {
        "fixed_acidity": 7897987,
        "volatile_acidity": 555,
        "citric_acid": 99,
        "residual_sugar": 99,
        "chlorides": 12,
        "free_sulfur_dioxide": 789,
        "total_sulfur_dioxide": 75,
        "density": 2,
        "pH": 3,
        "sulphates": 9,
        "alcohol": 9
    },
    "correct Range": {
        "fixed_acidity": 5,
        "volatile_acidity": 1,
        "citric_acid": 0.5,
        "residual_sugar": 10,
        "chlorides": 0.5,
        "free_sulfur_dioxide": 3,
        "total_sulfur_dioxide": 75,
        "density": 1,
        "pH": 3,
        "sulphates": 1,
        "alcohol": 9
    },
    "Incorrect Col": {
        "wrong_column": 5,
        "volatile_acidity": 1,
        "citric_acid": 0.5,
        "residual_sugar": 10,
        "chlorides": 0.5,
        "free_sulfur_dioxide": 3,
        "total_sulfur_dioxide": 75,
        "density": 1,
        "pH": 3,
        "sulphates": 1,
        "alcohol": 9
    }
}


TARGET_range = {
    "min": 3.0,
    "max": 8.0
}


def test_form_response_correct_range():
    data = input_data["correct Range"]
    res = form_response(data)
    assert TARGET_range["min"] <= res <= TARGET_range["max"]


def test_api_response_correct_range():
    data = input_data["correct Range"]
    res = api_response(data)
    assert TARGET_range["min"] <= res["response"] <= TARGET_range["max"]


def test_form_response_incorrect_range():
    data = input_data["Incorrect Range"]
    with pytest.raises(prediction_service.prediction.NotInRange):
        form_response(data)


def test_api_response_incorrect_range():
    data = input_data["Incorrect Range"]
    res = api_response(data)
    assert "should be between" in res["response"]


def test_api_response_incorrect_col():
    data = input_data["Incorrect Col"]
    res = api_response(data)
    assert "wrong_column is not a valid column" == res["response"]