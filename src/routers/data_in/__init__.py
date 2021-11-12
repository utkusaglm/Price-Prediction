from fastapi import APIRouter
from typing import Final, Tuple
from . import data_in
import logging

ROUTERS: Final[Tuple[APIRouter, ...]] = (
    data_in.router,
)