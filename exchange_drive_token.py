import json
import urllib.request
import urllib.parse

CLIENT_CFG_PATH = "/home/ubuntu/.gdrive_oauth_client.json"
CODE_PATH = "/home/ubuntu/.gdrive_auth_code.txt"
TOKEN_OUT_PATH = "/home/ubuntu/.gdrive_token.json"

# This is the redirect_uri actually used for this particular
# authorization request (OAuth Playground), which must match
# exactly what Google issued the code against.
REDIRECT_URI = "https://developers.google.com/oauthplayground"

with open(CLIENT_CFG_PATH) as f:
    cfg = json.load(f)

with open(CODE_PATH) as f:
    code = f.read().strip()

data = urllib.parse.urlencode({
    "code": code,
    "client_id": cfg["client_id"],
    "client_secret": cfg["client_secret"],
    "redirect_uri": REDIRECT_URI,
    "grant_type": "authorization_code",
}).encode()

req = urllib.request.Request(
    cfg["token_uri"],
    data=data,
    headers={"Content-Type": "application/x-www-form-urlencoded"},
)

try:
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode()
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print("TOKEN EXCHANGE FAILED")
    print(body)
    raise SystemExit(1)

token = json.loads(body)

if "refresh_token" not in token:
    print("WARNING: no refresh_token in response (already consumed once before?):")
    print(json.dumps(token, indent=2))
    raise SystemExit(1)

with open(TOKEN_OUT_PATH, "w") as f:
    json.dump(token, f, indent=2)

import os
os.chmod(TOKEN_OUT_PATH, 0o600)

print("SUCCESS")
print("Scopes granted:", token.get("scope"))
print("Access token expires in:", token.get("expires_in"), "seconds")
print("Refresh token saved:", "yes" if token.get("refresh_token") else "no")
