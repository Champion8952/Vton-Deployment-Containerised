import uvicorn
import asyncio
from server import app
from config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level="info",
        reload=True,
        reload_delay=0.25,
        reload_dirs=["./"],
        reload_includes=["*.py", "*.html", "*.css", "*.js"],
        workers=1
    )   
    
    # # Run the server in the main thread
    # server.run()
