CREATE TABLE
  RetryTable ( CaseSetId STRING(128) NOT NULL,
    RunId STRING(128) NOT NULL,
    CaseId INT64 NOT NULL,
    n_retried INT64,
    )
PRIMARY KEY
  (CaseSetId,
    RunId,
    CaseId);