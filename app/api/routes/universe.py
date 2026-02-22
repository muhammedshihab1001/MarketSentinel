import logging
from fastapi import APIRouter

from core.market.universe import MarketUniverse

router = APIRouter()
logger = logging.getLogger("marketsentinel.universe")


@router.get("/universe")
def universe_info():

    snapshot = MarketUniverse.snapshot()

    return {
        "version": snapshot.get("version"),
        "description": snapshot.get("description"),
        "tickers": snapshot.get("tickers"),
        "count": len(snapshot.get("tickers", [])),
        "universe_hash": snapshot.get("universe_hash")
    }