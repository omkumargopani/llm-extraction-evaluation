import re
from src.schema import OrderExtraction

FIELDS = ("customer_name", "product", "amount")

def normalize(value):
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return value

def field_accuracy(expected: dict, predicted: dict) -> float:
    return sum(normalize(expected.get(k)) == normalize(predicted.get(k)) for k in FIELDS) / len(FIELDS)

def field_results(expected: dict, predicted: dict) -> dict[str, bool]:
    return {key: normalize(expected.get(key)) == normalize(predicted.get(key)) for key in FIELDS}

def schema_valid(predicted: dict) -> bool:
    try:
        OrderExtraction.model_validate(predicted, strict=True)
        return True
    except (ValueError, TypeError):
        return False

def grounding_score(text: str, predicted: dict) -> float:
    normalized_text = " ".join(text.casefold().split())
    supported = []
    for key in FIELDS:
        value = predicted.get(key)
        if value is None:
            continue
        if key == "amount" and isinstance(value, int) and not isinstance(value, bool):
            chunks = re.findall(r"\d[\d,]*", text)
            supported.append(any(int(chunk.replace(",", "")) == value for chunk in chunks))
        elif isinstance(value, str):
            if key == "product" and " and " in value.casefold():
                supported.append(all(part.strip() in normalized_text for part in value.casefold().split(" and ")))
            else:
                supported.append(normalize(value) in normalized_text)
        else:
            supported.append(False)
    return sum(supported) / len(supported) if supported else 1.0

def error_categories(expected: dict, predicted: dict, valid: bool, grounding: float) -> list[str]:
    errors = []
    if not valid:
        errors.append("schema_invalid")
    for key in FIELDS:
        expected_value = expected.get(key)
        predicted_value = predicted.get(key)
        if expected_value is not None and predicted_value is None:
            errors.append(f"missing_{key}")
        elif expected_value is None and predicted_value is not None:
            errors.append(f"unsupported_{key}")
        elif normalize(expected_value) != normalize(predicted_value):
            errors.append(f"incorrect_{key}")
    if grounding < 1:
        errors.append("possible_ungrounded_value")
    return errors
