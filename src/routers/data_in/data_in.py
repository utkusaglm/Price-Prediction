from typing import *
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from uvicorn.main import main

router = APIRouter(prefix="/data_in")
TAG=["data_in"]

@router.get("/service_ok", tags = TAG, status_code=status.HTTP_200_OK)
async def get_measurement():
    return "ok"

