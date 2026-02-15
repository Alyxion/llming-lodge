"""Test package for llming-lodge."""
import os

# ensure environment variables are set
from dotenv import load_dotenv

dir = os.path.dirname(__file__)

max_step = 5
# find the .env file
cur_step = 0
while not os.path.exists(os.path.join(dir, ".env")):
    dir = os.path.dirname(dir)
    if dir == "/":
        raise Exception("Could not find .env file")
    cur_step += 1
    if cur_step > max_step:
        raise Exception("Could not find .env file")

load_dotenv(os.path.join(dir, ".env"))
