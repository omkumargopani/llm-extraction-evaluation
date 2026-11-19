import os
from openai import OpenAI
from src.schema import OrderExtraction

COMMON = (
    "Extract the buyer name, purchased product, and final amount paid only from a completed purchase. "
    "A completed purchase must be explicitly confirmed by the source text. "
    "If the transaction is cancelled, failed, rejected, pending, only planned, or not completed, "
    "return null for customer_name, product, and amount. "
    "Do not extract customer names, products, or amounts from incomplete transactions. "
    "If there is insufficient evidence of a completed purchase, return null for all three fields. "
    "Use null for individual details that are not stated in a confirmed completed purchase. "
    "If the source gives both list price and final paid amount, use the final paid amount. "
    "If shipping is included in a stated grand total, use the grand total. "
    "Do not mistake a seller for a buyer. "
    "For multiple products purchased together, combine their names using ' and '. "
    "Preserve the source wording of buyer and product names, and do not invent details."
)
PROMPTS = {
    "zero_shot": COMMON,
    "few_shot": COMMON + (
        " Examples: 'Rina bought a tablet for Rs. 20000.' => "
        "{customer_name: Rina, product: tablet, amount: 20000}. "
        "'Maya offered a camera for Rs. 7000 but nobody bought it.' => "
        "{customer_name: null, product: null, amount: null}. "
        "'Aman sold a mouse to Dev for Rs. 500.' => "
        "{customer_name: Dev, product: mouse, amount: 500}."
    ),
}

def extract_order(text: str, approach: str = "zero_shot") -> dict:
    if approach not in PROMPTS:
        raise ValueError(f"Unknown approach: {approach}")
    client = OpenAI()
    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        input=[
            {"role": "system", "content": PROMPTS[approach]},
            {"role": "user", "content": text},
        ],
        text_format=OrderExtraction,
    )
    if response.output_parsed is None:
        raise ValueError("No parsed structured output returned")
    return response.output_parsed.model_dump()
