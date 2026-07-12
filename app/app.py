from shiny.express import input, render, ui
from shiny import reactive
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import torch.nn.functional as F

label_dict = {
    'AnnualCrop': 0, 'Forest': 1, 'HerbaceousVegetation': 2,
    'Highway': 3, 'Industrial': 4, 'Pasture': 5,
    'PermanentCrop': 6, 'Residential': 7, 'River': 8, 'SeaLake': 9
}
r_label_dict = {v: k for k, v in label_dict.items()}

device = torch.device("cpu")

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        out = out + x
        out = self.final_relu(out)
        return out

class SatelliteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(32),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128)
        )
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.flatten(x)
        x = self.linear_layer(x)
        return x

# Load model and weights
model = SatelliteCNN().to(device)
state_dict = torch.load("assets/weights/best_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# App UI
ui.page_opts(title="Satellite Image Classifier Dashboard", fillable=True)

with ui.sidebar():
    ui.h5("Data Input")
    ui.input_file(
        "image_upload",
        "Select satellite image (PNG/JPG):",
        accept=[".jpg", ".jpeg", ".png"],
        multiple=False
    )    
    ui.input_action_button("predict_btn", "Run Inference", class_="btn-success w-100")

@reactive.calc
@reactive.event(input.predict_btn)
def prediction():
    file_infos = input.image_upload()
    if not file_infos:
        return None

    img_path = file_infos[0]["datapath"]
    img = Image.open(img_path).convert("RGB")
    tensor_img = image_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_outputs = model(tensor_img)
        probabilities = F.softmax(raw_outputs, dim=1)[0] * 100

    results = []
    for i, prob in enumerate(probabilities):
        results.append({
            "Class": r_label_dict[i],
            "Confidence": prob.item()
        })

    df = pd.DataFrame(results).sort_values(
        by="Confidence", ascending=False
    ).reset_index(drop=True)

    return {
        "df": df,
        "top_class": df.iloc[0]["Class"],
        "top_conf": df.iloc[0]["Confidence"],
    }

with ui.layout_columns(col_widths=(5, 7)):

    with ui.card():
        ui.card_header("Target Image Preview")

        @render.image
        def show_image():
            file_infos = input.image_upload()
            if not file_infos:
                return None

            return {
                "src": file_infos[0]["datapath"],
                "style": (
                    "width: 100%; max-height: 300px; object-fit: contain; "
                    "border-radius: 6px; margin: 0 auto; display: block;"
                )
            }

    with ui.card():
        ui.card_header("Class Probabilities")

        @render.data_frame
        def show_predictions():
            result = prediction()
            if result is None:
                return render.DataGrid(pd.DataFrame(columns=["Class", "Probability"]))

            df = result["df"].copy()
            df["Probability"] = df["Confidence"].apply(lambda p: f"{p:.2f}%")
            return render.DataGrid(df[["Class", "Probability"]], row_selection_mode="none")

with ui.card():
    ui.card_header("Inference Results")

    with ui.value_box(show_full_screen=True):
        "Classification Outcome"
        
        @render.text
        def main_prediction_text():
            result = prediction()
            if result is None:
                return "Awaiting Image Ingestion..."
            return result["top_class"]

        @render.text
        def sub_prediction_text():
            result = prediction()
            if result is None:
                return "Upload a valid raster dataset from the sidebar to trigger neural model."
            return f"Model Prediction Confidence: {result['top_conf']:.2f}%"