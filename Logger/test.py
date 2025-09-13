import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

client = WebClient(token="")

try:
    resp = client.chat_postMessage(
        channel="",
        text="Hello world"
    )
    print("sent ts:", resp["ts"])
except SlackApiError as e:
    print("error:", e.response.get("error"))
