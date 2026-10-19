"""Investigate failed pipeline runs through the AWS SDK (boto3), read-only.

Entry points: :func:`strandline.aws.batch.investigate_batch` and :func:`strandline.aws.omics.investigate_omics`.
Every AWS call goes through :class:`strandline.aws.session.Clients`, so tests can swap in fake clients.
"""
