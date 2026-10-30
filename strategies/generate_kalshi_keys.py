"""
Generates a 2048-bit RSA key pair for Kalshi API authentication.

Run this once, then upload the PRINTED PUBLIC KEY to your Kalshi account
(demo.kalshi.co for paper trading, kalshi.com for real money) under
Settings -> API. Kalshi gives you back an API Key ID -- set that as
KALSHI_API_KEY_ID, and point KALSHI_PRIVATE_KEY_PATH at the private key
file this script writes.

Usage:
    pip install cryptography
    python generate_kalshi_keys.py --out kalshi_private.pem
"""

import argparse

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="kalshi_private.pem", help="Where to write the PRIVATE key. Keep this file secret -- never commit it.")
    args = p.parse_args()

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with open(args.out, "wb") as f:
        f.write(private_pem)

    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    print(f"Private key written to {args.out} -- keep this file secret, never commit it.\n")
    print("Upload this PUBLIC key at demo.kalshi.co (Settings -> API) to get your API Key ID:\n")
    print(public_pem.decode("utf-8"))


if __name__ == "__main__":
    main()
