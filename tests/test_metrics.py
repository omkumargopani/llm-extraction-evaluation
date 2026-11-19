import pytest
from src.metrics import field_accuracy, field_results, schema_valid, grounding_score, error_categories
from src.report import summarize

def test_accuracy_full():
    expected = {"customer_name": "Rahul", "product": "iPhone 15", "amount": 65000}
    assert field_accuracy(expected, expected) == 1.0

def test_accuracy_partial():
    expected = {"customer_name": "Rahul", "product": "iPhone 15", "amount": 65000}
    predicted = {"customer_name": "Rahul", "product": "iPhone 14", "amount": 65000}
    assert field_accuracy(expected, predicted) == pytest.approx(2 / 3)

def test_field_results():
    assert field_results({"customer_name":"A", "product": "B", "amount": 1}, {"customer_name": "A", "product": "C", "amount": 1}) == {"customer_name": True, "product": False, "amount": True}

def test_schema_valid():
    assert schema_valid({"customer_name": "Rahul", "product": "iPhone 15", "amount": 65000})

def test_schema_invalid():
    assert not schema_valid({"customer_name": "Rahul", "product": "iPhone 15", "amount": "wrong"})

def test_schema_extra_field():
    assert not schema_valid({"customer_name": "Rahul", "product": "iPhone 15", "amount": 65000, "city": "Delhi"})

def test_grounding():
    assert grounding_score("Rahul ordered an iPhone 15 for Rs. 65,000", {"customer_name": "Rahul", "product": "iPhone 15", "amount": 65000}) == 1.0

def test_ungrounded():
    assert grounding_score("Rahul ordered an iPhone 15 for Rs. 65000", {"customer_name": "Rahul", "product": "iPhone 14", "amount": 65000}) == pytest.approx(2 / 3)

def test_grounding_missing_values():
    assert grounding_score("No purchase was made", {"customer_name": None, "product": None, "amount": None}) == 1.0

def test_errors():
    expected = {"customer_name": "Rahul", "product": "iPhone 15", "amount": 65000}
    prediction = {"customer_name": "Rahul", "product": None, "amount": 65000}
    assert "missing_product" in error_categories(expected, prediction, True, 1.0)

def test_unsupported_error():
    expected = {"customer_name": None, "product": None, "amount": None}
    predicted = {"customer_name": "Imaginary buyer", "product": None, "amount": None}
    assert "unsupported_customer_name" in error_categories(expected, predicted, True, 0.0)

def test_report_summary():
    rows = [{"accuracy": "0.66667", "exact_document_match": "False", "schema_valid": "True", "grounding": "0.5", "latency_seconds": "1.2", "correct_customer_name": "True", "correct_product": "False", "correct_amount": "True", "errors": "incorrect_product|possible_ungrounded_value"}]
    stats, errors = summarize(rows)
    assert stats["documents"] == 1
    assert stats["product_accuracy"] == 0.0
    assert errors["incorrect_product"] == 1
