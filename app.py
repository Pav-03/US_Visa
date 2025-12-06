from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from us_visa.pipline.prediction_pipeline import USVisaData, USVisaClassifier
from us_visa.pipline.training_pipeline import TrainPipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(
        "usvisa.html",{"request": request, "context": "Rendering"}
    )


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = await request.form()
        
        us_visa_data = USVisaData(
            continent=form.get("continent"),
            education_of_employee=form.get("education_of_employee"),
            has_job_experience=form.get("has_job_experience"),
            requires_job_training=form.get("requires_job_training"),
            no_of_employees=int(form.get("no_of_employees")),
            region_of_employment=form.get("region_of_employment"),
            prevailing_wage=int(form.get("prevailing_wage")),
            unit_of_wage=form.get("unit_of_wage"),
            full_time_position=form.get("full_time_position"),
            yr_of_estab=int(form.get("yr_of_estab"))
        )
        
        us_visa_df = us_visa_data.get_data_as_dataframe()
        model_predictor = USVisaClassifier()
        value = model_predictor.predict(dataframe=us_visa_df)[0]
        
        status = "Certified" if value == 1 else "Denied"
        
        return templates.TemplateResponse(
            "usvisa.html",
            {"request": request, "context": status}
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
