from fastapi import FastAPI

print("Hello, FastAPI!")

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

