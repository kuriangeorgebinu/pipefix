"""Investigate failed pipeline runs through the AWS SDK (boto3), read-only.

Entry points: :func:`pipefix.aws.batch.investigate_batch` and :func:`pipefix.aws.omics.investigate_omics`.
Every AWS call goes through :class:`pipefix.aws.session.Clients`, so tests can swap in fake clients.
"""
