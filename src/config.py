import os 
# Silence TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=all, 1=info, 2=warning, 3=error
import tensorflow as tf
from  dotenv import load_dotenv
import joblib
tf.get_logger().setLevel('ERROR')
# Load the environment variables from the .env file
load_dotenv()

# Get the variables
APP_NAME = os.getenv("APP_NAME")
VERSION = os.getenv("VERSION")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# Parent folder path
SRC_FOLDER_PATH=os.path.dirname(os.path.abspath(__file__))

# Downloaded images folder path
DOWNLOADED_IMAGES_FOLDER = os.path.join(SRC_FOLDER_PATH, "assets", "downloaded-images")
os.makedirs(DOWNLOADED_IMAGES_FOLDER, exist_ok=True)

# Load Model
MODEL = tf.keras.models.load_model(os.path.join(SRC_FOLDER_PATH, "assets", "model.keras"))
IDX2LABEL = joblib.load(os.path.join(SRC_FOLDER_PATH, "assets", "idx2label.joblib"))