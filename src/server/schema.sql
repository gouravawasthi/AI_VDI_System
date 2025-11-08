-- FILE: schema.sql

-- Set PRAGMA to optimize transaction speed (optional but good practice)
PRAGMA synchronous = OFF;

-- Table for Chip Inspection Data
CREATE TABLE IF NOT EXISTS CHIPINSPECTION 
    (Barcode TEXT NOT NULL,
     DT TEXT,             -- TEXT for DATETIME in SQLite
     Process_id INTEGER, 
     Station_ID INTEGER,  
     PASS_FAIL INTEGER);  -- INTEGER for BOOLEAN (1=PASS/TRUE, 0=FAIL/FALSE)
CREATE INDEX ChipInspectionByBarcodeIndex ON CHIPINSPECTION (Barcode);

-- ---------------------------------------------------------------------------------------------------

-- Table for Inline Inspection (Bottom) Data
CREATE TABLE IF NOT EXISTS INLINEINSPECTIONBOTTOM 
    (Barcode TEXT NOT NULL,
     DT TEXT,
     Process_id INTEGER, 
     Station_ID INTEGER,
     Antenna INTEGER,
     Capacitor INTEGER,
     Speaker INTEGER,
     Result INTEGER,
     ManualAntenna INTEGER,
     ManualCapacitor INTEGER,
     ManualSpeaker INTEGER,
     ManualResult INTEGER
     );
CREATE INDEX InlineBottomByBarcodeIndex ON INLINEINSPECTIONBOTTOM (Barcode);

-- ---------------------------------------------------------------------------------------------------

-- Table for Inline Inspection (Top) Data
CREATE TABLE IF NOT EXISTS INLINEINSPECTIONTOP
    (Barcode TEXT NOT NULL,
     DT TEXT,
     Process_id INTEGER, 
     Station_ID INTEGER,
     Screw INTEGER,
     Plate INTEGER,
     Result INTEGER,
     ManualScrew INTEGER,        
     ManualPlate INTEGER,  
     ManualResult INTEGER
    );
CREATE INDEX InlineTopByBarcodeIndex ON INLINEINSPECTIONTOP (Barcode);

-- ---------------------------------------------------------------------------------------------------

-- Table for End-of-Line Test (EOLT) Inspection Data
CREATE TABLE IF NOT EXISTS EOLTINSPECTION
    (Barcode TEXT NOT NULL,
     DT TEXT,
     Process_id INTEGER, 
     Station_ID INTEGER,
     Upper INTEGER,
     Lower INTEGER,
     Left INTEGER,
     Right INTEGER,        
     Result INTEGER,
     Printtext TEXT,
     Barcodetext TEXT,
     ManualUpper INTEGER,
     ManualLower INTEGER,
     ManualLeft INTEGER,
     ManualRight INTEGER,
     ManualResult INTEGER
    );
CREATE INDEX EOLTByBarcodeIndex ON EOLTINSPECTION (Barcode);