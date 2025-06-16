-- Create the database
CREATE DATABASE TelecomAnalytic;
GO

USE TelecomAnalytics;
GO

-- Create main table structure
CREATE TABLE UserSessions (
    BearerId BIGINT,
    StartTime DATETIME,
    StartMs INT,
    EndTime DATETIME,
    EndMs INT,
    DurationMs INT,
    IMSI BIGINT,
    MSISDN BIGINT,
    IMEI BIGINT,
    LastLocation NVARCHAR(100),
    AvgRttDL FLOAT,
    AvgRttUL FLOAT,
    AvgBearerTPDL FLOAT,
    AvgBearerTPUL FLOAT,
    TcpDLRetransVol INT,
    TcpULRetransVol INT,
    DLTPUnder50 FLOAT,
    DLTP50to250 FLOAT,
    DLTP250to1M FLOAT,
    DLTPOver1M FLOAT,
    ULTPUnder10 FLOAT,
    ULTP10to50 FLOAT,
    ULTP50to300 FLOAT,
    ULTPOver300 FLOAT,
    HttpDL INT,
    HttpUL INT,
    ActivityDurDL INT,
    ActivityDurUL INT,
    HandsetManufacturer NVARCHAR(100),
    HandsetType NVARCHAR(100),
    SocialMediaDL BIGINT,
    SocialMediaUL BIGINT,
    GoogleDL BIGINT,
    GoogleUL BIGINT,
    EmailDL BIGINT,
    EmailUL BIGINT,
    YoutubeDL BIGINT,
    YoutubeUL BIGINT,
    NetflixDL BIGINT,
    NetflixUL BIGINT,
    GamingDL BIGINT,
    GamingUL BIGINT,
    OtherDL BIGINT,
    OtherUL BIGINT,
    TotalUL BIGINT,
    TotalDL BIGINT
);
GO


-- Bulk insert from CSV
BULK INSERT UserSessions
FROM 'dataset\telcom_data (2).xlsx - Sheet1.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
);
GO


-- Top 10 handsets
SELECT TOP 10 
    HandsetType, 
    COUNT(*) AS UserCount
FROM UserSessions
WHERE HandsetType IS NOT NULL AND HandsetType != 'undefined'
GROUP BY HandsetType
ORDER BY UserCount DESC;

-- Top manufacturers with market share
WITH ManufacturerCounts AS (
    SELECT 
        HandsetManufacturer,
        COUNT(*) AS TotalUsers,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()) AS MarketShare
    FROM UserSessions
    WHERE HandsetManufacturer IS NOT NULL
    GROUP BY HandsetManufacturer
)
SELECT TOP 5 *
FROM ManufacturerCounts
ORDER BY TotalUsers DESC;


-- User engagement metrics
SELECT 
    MSISDN AS UserID,
    COUNT(*) AS SessionCount,
    SUM(DurationMs)/3600000.0 AS TotalHours,
    SUM(TotalDL + TotalUL) AS TotalDataBytes,
    SUM(TotalDL + TotalUL)/POWER(1024,3) AS TotalDataGB
FROM UserSessions
GROUP BY MSISDN
ORDER BY TotalDataBytes DESC;


-- Network experience metrics
SELECT 
    MSISDN AS UserID,
    HandsetType,
    AVG(AvgRttDL) AS AvgRtt,
    AVG(AvgBearerTPDL) AS AvgThroughput,
    SUM(ISNULL(TcpDLRetransVol,0)) AS TotalRetransmissions
FROM UserSessions
GROUP BY MSISDN, HandsetType
ORDER BY AvgThroughput DESC;

-- Satisfaction scoring (conceptual)
WITH EngagementScores AS (
    SELECT 
        MSISDN,
        (SessionCount - MIN(SessionCount) OVER ()) * 1.0 / 
            (MAX(SessionCount) OVER () - MIN(SessionCount) OVER ()) AS NormalizedEngagement
    FROM (
        SELECT MSISDN, COUNT(*) AS SessionCount
        FROM UserSessions
        GROUP BY MSISDN
    ) t
),
ExperienceScores AS (
    SELECT 
        MSISDN,
        1.0 - ((AvgRtt - MIN(AvgRtt) OVER ()) / 
              (MAX(AvgRtt) OVER () - MIN(AvgRtt) OVER ())) AS NormalizedExperience
    FROM (
        SELECT MSISDN, AVG(AvgRttDL) AS AvgRtt
        FROM UserSessions
        GROUP BY MSISDN
    ) t
)
SELECT 
    e.MSISDN AS UserID,
    e.NormalizedEngagement,
    x.NormalizedExperience,
    (e.NormalizedEngagement + x.NormalizedExperience) / 2 AS SatisfactionScore
FROM EngagementScores e
JOIN ExperienceScores x ON e.MSISDN = x.MSISDN
ORDER BY SatisfactionScore DESC;


-- Create indexes for performance
CREATE INDEX IX_UserSessions_MSISDN ON UserSessions(MSISDN);
CREATE INDEX IX_UserSessions_Handset ON UserSessions(HandsetManufacturer, HandsetType);
CREATE INDEX IX_UserSessions_Network ON UserSessions(AvgRttDL, AvgBearerTPDL);
GO

-- Create view for common aggregations
CREATE VIEW UserEngagementSummary AS
SELECT 
    MSISDN AS UserID,
    COUNT(*) AS SessionCount,
    SUM(DurationMs) AS TotalDuration,
    SUM(TotalDL + TotalUL) AS TotalDataVolume,
    AVG(AvgBearerTPDL) AS AvgThroughput
FROM UserSessions
GROUP BY MSISDN;
GO


-- Procedure for manufacturer analysis
CREATE PROCEDURE sp_ManufacturerAnalysis
AS
BEGIN
    -- Market share
    SELECT 
        HandsetManufacturer,
        COUNT(*) AS UserCount,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM UserSessions) AS MarketSharePercent
    FROM UserSessions
    GROUP BY HandsetManufacturer
    ORDER BY UserCount DESC;
    
    -- Data usage by manufacturer
    SELECT 
        HandsetManufacturer,
        SUM(TotalDL)/POWER(1024,3) AS DownloadGB,
        SUM(TotalUL)/POWER(1024,3) AS UploadGB
    FROM UserSessions
    GROUP BY HandsetManufacturer
    ORDER BY DownloadGB DESC;
END;
GO

-- Procedure for application usage
CREATE PROCEDURE sp_ApplicationUsage
    @UserID BIGINT = NULL
AS
BEGIN
    SELECT 
        ISNULL(@UserID, MSISDN) AS UserID,
        SUM(SocialMediaDL + SocialMediaUL)/POWER(1024,2) AS SocialMediaMB,
        SUM(GoogleDL + GoogleUL)/POWER(1024,2) AS GoogleMB,
        SUM(YoutubeDL + YoutubeUL)/POWER(1024,2) AS YoutubeMB,
        SUM(NetflixDL + NetflixUL)/POWER(1024,2) AS NetflixMB
    FROM UserSessions
    WHERE @UserID IS NULL OR MSISDN = @UserID
    GROUP BY MSISDN;
END;
GO


-- Run manufacturer analysis
EXEC sp_ManufacturerAnalysis;

-- Get application usage for specific user
EXEC sp_ApplicationUsage @UserID = 33664962239;

-- Get top 10 satisfied users
SELECT TOP 10 *
FROM (
    SELECT 
        e.MSISDN AS UserID,
        e.NormalizedEngagement,
        x.NormalizedExperience,
        (e.NormalizedEngagement + x.NormalizedExperience) / 2 AS SatisfactionScore
    FROM EngagementScores e
    JOIN ExperienceScores x ON e.MSISDN = x.MSISDN
) t
ORDER BY SatisfactionScore DESC;