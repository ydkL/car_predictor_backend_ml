from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import firstTry
import json
import httpx
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)



@app.get("/process_data/")
async def process_data(param1: float, param2: float, param3: float, param4: float, param5: float, param6: float
                       , param7: float, param8: float, param9: float):
    """
    'Grid', 'Comfort', 'Tech', 'Visualize', 'Volume', 'Reliability', 'Security', 'Service', "Insulation"
    """

    # easy to use
    input_values = [param1, param2, param3, param4, param5, param6, param7, param8, param9]

    # Predict car model

    # Call another endpoint
    async with httpx.AsyncClient() as client:
        response = await client.get("https://car-predictor-backend-db.onrender.com/model-stats/")
        models = []
        model_names = []
        grid = []
        comfort = []
        tech = []
        visualize = []
        volume = []
        reliability = []
        security = []
        service = []
        insulation = []
        for model in response.json():
            model_names.append(model['modelName'])
            grid .append(model['Grid'])
            comfort.append(model['Comfort'])
            tech.append(model['Tech'])
            visualize.append(model['Visualize'])
            volume.append(model['Volume'])
            reliability.append(model['Reliability'])
            security.append(model['Security'])
            service.append(model['Service'])
            insulation.append(model['Insulation'])
            
        
        models.append({
            "Grid" : grid,
            "Models" : model_names
        })

        models.append({
            "Comfort": grid,
            "Models": model_names
        })

        models.append({
            "Tech": grid,
            "Models": model_names
        })

        models.append({
            "Visualize": grid,
            "Models": model_names
        })

        models.append({
            "Volume": grid,
            "Models": model_names
        })

        models.append({
            "Reliability": grid,
            "Models": model_names
        })

        models.append({
            "Security": grid,
            "Models": model_names
        })

        models.append({
            "Service": grid,
            "Models": model_names
        })

        models.append({
            "Insulation": grid,
            "Models": model_names
        })
    predicted_model = ""
    
    clf, le, X = firstTry.train_models(models)
    predicted_model = firstTry.predict_car_model(input_values, clf, le, X)
    print("Predicted car model easy to use:", predicted_model)
    
    # Perform processing here
    result = {
        "Predicted Model": predicted_model
    }
    json_result = json.dumps(result)
    return Response(content=json_result, media_type="application/json")
