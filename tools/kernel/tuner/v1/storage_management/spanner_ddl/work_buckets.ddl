CREATE TABLE
  WorkBuckets ( ID STRING(128) NOT NULL,
    RunId STRING(128) NOT NULL,
    BucketId INT64 NOT NULL,
    StartCaseId INT64 NOT NULL,
    EndCaseId INT64 NOT NULL,
    Status STRING(32),
    WorkerID STRING(128),
    TotalTime INT64,
    UpdatedAt TIMESTAMP OPTIONS ( allow_commit_timestamp = TRUE ),
    TPU STRING(32),
    )
PRIMARY KEY
  (ID,
    RunId,
    BucketId),
  INTERLEAVE IN PARENT CaseSet
ON
DELETE
  CASCADE;
