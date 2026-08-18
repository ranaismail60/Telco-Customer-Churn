-- ============================================================
-- NexaTel Churn Project — Database Schema
-- Normalized to 3rd Normal Form: one flat CSV -> 4 related tables
-- ============================================================

DROP TABLE IF EXISTS churn_status;
DROP TABLE IF EXISTS services;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS customers;

-- Core demographic info per customer
CREATE TABLE customers (
    customer_id     TEXT PRIMARY KEY,
    gender          TEXT,
    senior_citizen  INTEGER,   -- 0/1
    partner         TEXT,      -- Yes/No
    dependents      TEXT,      -- Yes/No
    tenure          INTEGER    -- months with company
);

-- Billing / contract info per customer
CREATE TABLE accounts (
    customer_id       TEXT PRIMARY KEY,
    contract          TEXT,    -- Month-to-month / One year / Two year
    paperless_billing TEXT,    -- Yes/No
    payment_method    TEXT,
    monthly_charges   REAL,
    total_charges     REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Subscribed services per customer
CREATE TABLE services (
    customer_id        TEXT PRIMARY KEY,
    phone_service       TEXT,
    multiple_lines       TEXT,
    internet_service    TEXT,  -- DSL / Fiber optic / No
    online_security      TEXT,
    online_backup         TEXT,
    device_protection     TEXT,
    tech_support          TEXT,
    streaming_tv           TEXT,
    streaming_movies       TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Target variable, kept in its own table on purpose:
-- keeps "the answer" physically separate from the feature tables,
-- which makes leakage mistakes (accidentally joining it into
-- training features) much easier to spot during code review.
CREATE TABLE churn_status (
    customer_id TEXT PRIMARY KEY,
    churn       TEXT,  -- Yes/No
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE INDEX idx_accounts_contract ON accounts(contract);
CREATE INDEX idx_services_internet ON services(internet_service);
