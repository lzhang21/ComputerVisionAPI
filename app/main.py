from fastapi import FastAPI
from app import classification

app = FastAPI()


@app.get("/") # path goes here / operation type
def root():
    return {"message": "Hello World"} #dict, list, singular values as str, int


# reading in file paths to images
@app.get("/model/")
def read_file():
    return {"message": "hello model"} #dict, list, singular values as str, int


@app.get("/model/{confidence}/url/{file_path:path}")

def read_user_item(confidence: int, file_path):
    output = classification.inference(confidence,file_path)
    return(output)



# https://img.freepik.com/premium-photo/set-casual-people-white_394555-1982.jpg
# https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/P3030027ParkingLot_wb.jpg/220px-P3030027ParkingLot_wb.jpg