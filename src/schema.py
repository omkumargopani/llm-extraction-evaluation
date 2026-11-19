from pydantic import BaseModel, ConfigDict

class OrderExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_name: str | None
    product: str | None
    amount: int | None
