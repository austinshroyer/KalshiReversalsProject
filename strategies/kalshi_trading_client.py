"""
Authenticated Kalshi trading client: balance, positions, and order
placement/cancellation against the demo (or production) Trade API V2.

Unlike kalshi_client.py -- used for the historical backfill, and read-only /
unauthenticated -- everything here requires a real API key + RSA private
key, because placing orders and reading your portfolio are private actions.
Kept as a separate module on purpose: the backfill/analysis side of this
project never needs write access to your account, so it never imports this
file.

SETUP:
  1. Create a demo account at https://demo.kalshi.co/ (mock personal info
     is fine and recommended -- see Kalshi's own demo account guide).
  2. Generate a key pair: python generate_kalshi_keys.py
  3. Upload the printed PUBLIC key at Settings -> API on your demo account.
     Kalshi gives you back an API Key ID.
  4. Set environment variables (don't hardcode these anywhere):
       export KALSHI_API_KEY_ID=<the id Kalshi gave you>
       export KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_private.pem
       export KALSHI_BASE_URL=https://external-api.demo.kalshi.co/trade-api/v2
     Demo and production credentials are NOT interchangeable -- a demo key
     simply won't authenticate against the production host, but double
     check KALSHI_BASE_URL yourself before running anything that trades.

Order schema note: this targets Kalshi's current V2 order endpoint
(POST /portfolio/events/orders), which quotes everything from the YES leg:
side="bid" means buy YES, side="ask" means sell YES (selling YES is how you
close/cash out a YES position you hold). The older /portfolio/orders
endpoint (buy/sell + yes/no + integer-cent prices) is being phased out and
is deliberately not used here.
"""

import base64
import time
import uuid

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

import config


class KalshiTradingError(RuntimeError):
    pass


class KalshiTradingClient:
    def __init__(self, base_url: str = None, api_key_id: str = None, private_key_path: str = None):
        self.base_url = (base_url or config.KALSHI_BASE_URL).rstrip("/")
        self.api_key_id = api_key_id or config.KALSHI_API_KEY_ID
        self.private_key_path = private_key_path or config.KALSHI_PRIVATE_KEY_PATH
        if not self.api_key_id or not self.private_key_path:
            raise KalshiTradingError(
                "Trading requires KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH (env vars or "
                "config.py). Run generate_kalshi_keys.py to create a key pair, then register the "
                "public key at https://demo.kalshi.co/ (Settings -> API)."
            )
        with open(self.private_key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        self.session = requests.Session()

    def _headers(self, method: str, path: str) -> dict:
        timestamp_ms = str(int(time.time() * 1000))
        signed_path = f"/trade-api/v2{path}"
        message = f"{timestamp_ms}{method.upper()}{signed_path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, params: dict = None, json_body: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._headers(method, path)
        resp = self.session.request(
            method, url, headers=headers, params=params, json=json_body, timeout=config.REQUEST_TIMEOUT_SECONDS
        )
        if resp.status_code >= 400:
            raise KalshiTradingError(f"{method} {path} -> HTTP {resp.status_code}: {resp.text}")
        if resp.status_code == 204 or not resp.text:
            return {}
        return resp.json()

    # -- portfolio ----------------------------------------------------------

    def get_balance(self) -> dict:
        return self._request("GET", "/portfolio/balance")

    def get_positions(self) -> dict:
        return self._request("GET", "/portfolio/positions")

    # -- orders ---------------------------------------------------------------

    def create_order(
        self,
        ticker: str,
        side: str,  # "bid" = buy YES, "ask" = sell YES
        count: float,
        price: float,  # 0-1 dollars
        time_in_force: str = "immediate_or_cancel",
        reduce_only: bool = False,
        client_order_id: str = None,
    ) -> dict:
        if side not in ("bid", "ask"):
            raise ValueError("side must be 'bid' (buy YES) or 'ask' (sell YES)")
        price = max(0.01, min(0.99, price))
        payload = {
            "ticker": ticker,
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "side": side,
            "count": f"{count:.2f}",
            "price": f"{price:.4f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
            "reduce_only": reduce_only,
        }
        return self._request("POST", "/portfolio/events/orders", json_body=payload)

    def cancel_order(self, order_id: str) -> dict:
        return self._request("DELETE", f"/portfolio/events/orders/{order_id}")
