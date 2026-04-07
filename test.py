import pydantic


class User(pydantic.BaseModel):
    name: str
    age: int