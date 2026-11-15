CREATE TABLE KernelAutoTuneCases (
    CaseSetId STRING(128) NOT NULL,
    CaseId STRING(36) DEFAULT (GENERATE_UUID()),
    CaseKeyValue STRING(1024),
    KernelTunerName STRING(64) NOT NULL,
    TPU STRING(32) NOT NULL,
) PRIMARY KEY (CaseSetId, CaseId);
