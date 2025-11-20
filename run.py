import os
import uvicorn
from api import app
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Get the port Render assigns (default to 8000 if not set)
    port = int(os.environ.get("PORT", 8000))
    
    # Run the app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)