CREATE TABLE
  CaseResults ( ID STRING(128) NOT NULL,
    RunId STRING(128) NOT NULL,
    CaseId INT64 NOT NULL,
    ProcessedStatus STRING(32),
    WorkerID STRING(128),
    Latency INT64,
    WarmupTime INT64,
    TotalTime INT64,
    ProcessedAt TIMESTAMP OPTIONS ( allow_commit_timestamp = TRUE ),
    TPU STRING(32),
    )
PRIMARY KEY
  (ID,
    RunId,
    CaseId),
  INTERLEAVE IN PARENT CaseSet
ON
DELETE
  CASCADE;
