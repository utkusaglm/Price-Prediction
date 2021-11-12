import config
import uvicorn
from fastapi.routing import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from routers import data_in
from fastapi import FastAPI, Request, status
import traceback
from typing import *
ROUTERS: Final[Tuple[APIRouter, ...]] = (*data_in.ROUTERS,)

app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
for router in ROUTERS:
    app.include_router(router)

def serve() -> None:
    uvicorn.run(app, host=config.HOST, port=config.PORT)