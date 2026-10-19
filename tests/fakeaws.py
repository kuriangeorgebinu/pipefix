"""Test helpers: the fake AWS account lives in strandline.demo.fake so the UI demo can use it too."""
from strandline.demo.fake import *  # noqa: F401,F403
from strandline.demo.fake import ClientError, fake_clients  # noqa: F401
