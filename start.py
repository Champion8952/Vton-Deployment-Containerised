from waitress import serve
from server import app
from config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    serve(
        app, 
        host="0.0.0.0",
        port=settings.PORT,
        threads=30,
        connection_limit=1000,
        channel_timeout=300,
        ident="TryOn Server"
        )
