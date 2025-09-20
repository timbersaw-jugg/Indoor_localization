# visualize.py

import pandas as pd
import numpy as np
from src.utils import animate_paths, load_config
from src.data_processor import DataProcessor

def main():
    """
    Loads the final predictions and generates an animated visualization.
    """
    print("Loading configuration and data for visualization...")
    config = load_config()

    # We need the idx_to_coords map, which we can get by re-running part of the data processing
    data_processor = DataProcessor(config)
    # --- THIS IS THE FIX ---
    # Call the correct function name and unpack the correct number of return values
    _, _, idx_to_coords = data_processor.load_and_prepare_data()
    # --- END OF FIX ---

    # Load the results saved by run.py
    try:
        results_df = pd.read_csv('final_predictions.csv')
        final_preds = results_df['predicted_class'].values
        y_test = results_df['true_class'].values
    except FileNotFoundError:
        print("Error: `final_predictions.csv` not found.")
        print("Please run `run.py` first to generate the prediction results.")
        return

    print("Converting predictions to coordinates...")
    pred_coords = [idx_to_coords[int(idx)] for idx in final_preds]
    true_coords = [idx_to_coords[int(idx)] for idx in y_test]
    timestamps = pd.to_datetime(pd.date_range(start='2024-01-01', periods=len(y_test), freq='s'))

    BEACON_COORDS = {
        "b3001": (7, 13), "b3002": (18, 14), "b3003": (7, 3),  "b3004": (9, 8),
        "b3005": (4, 9),  "b3006": (16, 9), "b3007": (12, 14), "b3008": (21, 8),
        "b3009": (3, 6),  "b3010": (11, 4), "b3011": (17, 5),  "b3012": (14, 16), "b3013": (23, 4)
    }
    BEACON_ROOM_MAP = {
        "b3001": "Bedroom", "b3002": "Master Bedroom", "b3003": "Patio", "b3004": "Common Area",
        "b3005": "Left Balcony", "b3006": "Dining Hall", "b3007": "Rest Room", "b3008": "Right Balcony",
        "b3009": "Store Room", "b3010": "Entrance", "b3011": "Kitchen", "b3012": "Laundry", "b3013": "Dish Washing"
    }
    img_path = 'data/3d House.jpg' # Make sure this image is in your root folder

    print("Generating animation... This may take a moment.")
    anim = animate_paths(
        true_coords=true_coords,
        pred_coords=pred_coords,
        timestamps=timestamps,
        image_path=img_path,
        xrange=(0, 25), yrange=(0, 20),
        beacon_coords=BEACON_COORDS,
        beacon_names=BEACON_ROOM_MAP,
        title='Indoor Position Tracking: Actual vs Predicted Paths'
    )

    # Save the animation as a video file
    anim.save('final_animation.mp4', writer='ffmpeg', dpi=150)
    print("\n✅ Animation saved to `final_animation.mp4`!")

if __name__ == "__main__":
    main()