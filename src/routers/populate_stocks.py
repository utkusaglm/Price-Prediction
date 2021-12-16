import psycopg2
import psycopg2.extras
import alpaca_trade_api as  tradeapi

class Config:
    DB_HOST ='localhost'
    DB_USER ='postgres'
    DB_PASS ='postgres'
    DB_NAME ='btc'
    API_URL = "https://paper-api.alpaca.markets"
    API_KEY  = "PKTE1915P0HLD2R6B8E6"
    API_SECRET ="OtIa186qJPsGCPzYTryYJI74dkMq5xEEhs0bJNZP"


config=Config()
# connection=psycopg2.connect(host=config.DB_HOST, database=config.DB_NAME, user= config.DB_USER, password=config.DB_PASS)
# cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

api = tradeapi.REST(config.API_KEY, config.API_SECRET, base_url=config.API_URL)
assets=api.get_crypto_bars("BTCUSD",tradeapi.rest.TimeFrame.Minute, "2021-06-08", "2021-06-08").df
# for asset in assets:
#     if "BTCUSD" in asset.name:
#         print(asset)
# barset  = api.get_bars("BTCUSD", tradeapi.rest.TimeFrame.Hour, "2021-06-08", "2021-06-08", adjustment='raw').df
# aapl_bars = barset
print(assets)
# for asset in assets:
#     print(f"Inserting stock {asset.name} {asset.symbol}")
#     # cursor.execute("""
#     #     INSERT INTO stock (name, symbol, exchange, is_etf) 
#     #     VALUES (%s, %s, %s, false)
#     # """, (asset.name, asset.symbol, asset.exchange))

# connection.commit()
