from pydantic import BaseModel


class AgeInput(BaseModel):
    Length: float
    Diameter: float
    Height: float
    Whole_weight: float
    Shucked_weight: float
    Viscera_weight: float
    Shell_weight: float
    Sex_F: bool
    Sex_I: bool
    Sex_M: bool


class AgeOutput(BaseModel):
    age: int
