CREATE TABLE
  CaseSet ( ID STRING(128) NOT NULL,
    Description STRING(1024),
    Status STRING(32),
    ScanSpace INT64,
    Valid INT64,
    Invalid INT64,
    DurationSeconds FLOAT64,
    )
PRIMARY KEY
  (ID);
