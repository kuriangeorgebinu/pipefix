"""Test helpers: the fake AWS account lives in pipefix.demo.fake so the UI demo can use it too."""
from pipefix.demo.fake import *  # noqa: F401,F403
from pipefix.demo.fake import ClientError, fake_clients  # noqa: F401
