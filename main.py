from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.models import Base, History
import os
import io
import time
import json
from DL_model import predict_image, init_model
from PIL import Image

from crud import get_request_stats

app = FastAPI()
UPLOAD_DIR = "files/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
model, device = init_model()

engine = create_engine('sqlite:///library.db')
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

TOKEN = "42bo$$42"

def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()

@app.get('/predict/')
async def predict(image: UploadFile = File(),
                  db: Session = Depends(get_db)):
    global model
    global device

    start_time = time.time()

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="bad request"
        )
    try:
        contents = await image.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="bad request"
            )

        image_bytes = io.BytesIO(contents)
        try:
            pil_image = Image.open(image_bytes).convert('RGB')
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="bad request"
            )

        try:
            prediction = predict_image(pil_image, model, device)
        except Exception as model_error:
            print(f"Model error: {str(model_error)}")
            raise HTTPException(
                status_code=403,
                detail="модель не смогла обработать данные"
            )

        if not prediction or len(prediction) == 0:
            raise HTTPException(
                status_code=403,
                detail="модель не смогла обработать данные"
            )

        image_info = {
            "filename": image.filename,
            "content_type": image.content_type,
            "original_size": pil_image.size,
            "file_size_bytes": len(contents)
        }

        result = {
            "success": True,
            "image_info": image_info,
            "top_predictions": prediction,
            "top_prediction": prediction[0] if prediction else None
        }

        history_record = History(
            request = json.dumps({'filename' : image.filename or "unknown",
                       'content_type' : image.content_type,
                        'original_width' : pil_image.width,
                        'original_height' : pil_image.height,
                        'file_size_bytes' :len(contents)}),
            response = json.dumps(result),
            processing_time = round((time.time() - start_time) * 1000, 2),
        )

        db.add(history_record)
        db.commit()

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(History).all()

@app.delete("/history")
def delete_history(token: str = Header(...),
                db : Session = Depends(get_db)):
    if token != TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    db.query(History).delete()
    db.commit()
    return {"message": "History cleared"}


@app.get("/stats")
async def stats(db: Session = Depends(get_db)):
    return await get_request_stats(db)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)