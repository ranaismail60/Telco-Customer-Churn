-- ============================================================
-- NexaTel Churn Project — Business SQL Queries
-- Run against data/nexatel.db (sqlite3 data/nexatel.db < sql/queries.sql)
-- ============================================================

-- Q1: What is the overall churn rate?
SELECT
    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM churn_status;

-- Q2: How does churn rate vary by contract type?
SELECT
    a.contract,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM accounts a
JOIN churn_status c ON a.customer_id = c.customer_id
GROUP BY a.contract
ORDER BY churn_rate_pct DESC;

-- Q3: How does churn rate vary by internet service type?
SELECT
    s.internet_service,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM services s
JOIN churn_status c ON s.customer_id = c.customer_id
GROUP BY s.internet_service
ORDER BY churn_rate_pct DESC;

-- Q4: Average tenure of churned vs. retained customers.
SELECT
    c.churn,
    ROUND(AVG(cu.tenure), 1) AS avg_tenure_months
FROM customers cu
JOIN churn_status c ON cu.customer_id = c.customer_id
GROUP BY c.churn;

-- Q5: Average monthly charges of churned vs. retained customers.
SELECT
    c.churn,
    ROUND(AVG(a.monthly_charges), 2) AS avg_monthly_charges
FROM accounts a
JOIN churn_status c ON a.customer_id = c.customer_id
GROUP BY c.churn;

-- Q6: Top 5 (contract x payment method) segments with highest churn rate,
-- restricted to segments with a meaningful sample size (>= 30 customers).
SELECT
    a.contract,
    a.payment_method,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM accounts a
JOIN churn_status c ON a.customer_id = c.customer_id
GROUP BY a.contract, a.payment_method
HAVING COUNT(*) >= 30
ORDER BY churn_rate_pct DESC
LIMIT 5;

-- Q7: Total monthly revenue currently at risk (sum of MonthlyCharges
-- for customers who have already churned — used as the "headline" dollar
-- figure for the VP of Retention).
SELECT
    ROUND(SUM(a.monthly_charges), 2) AS monthly_revenue_at_risk
FROM accounts a
JOIN churn_status c ON a.customer_id = c.customer_id
WHERE c.churn = 'Yes';

-- Q8: Churn rate for the highest-risk known segment: tenure < 6 months
-- AND no tech support.
SELECT
    COUNT(*) AS segment_size,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers cu
JOIN services s ON cu.customer_id = s.customer_id
JOIN churn_status c ON cu.customer_id = c.customer_id
WHERE cu.tenure < 6
  AND s.tech_support = 'No';

-- Q9: Does the number of subscribed add-on services relate to churn?
-- (counts Yes-valued add-ons per customer, then buckets and compares churn rate)
WITH service_counts AS (
    SELECT
        customer_id,
        (CASE WHEN online_security   = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN online_backup     = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN tech_support      = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN streaming_tv      = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN streaming_movies  = 'Yes' THEN 1 ELSE 0 END) AS num_addon_services
    FROM services
)
SELECT
    sc.num_addon_services,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM service_counts sc
JOIN churn_status c ON sc.customer_id = c.customer_id
GROUP BY sc.num_addon_services
ORDER BY sc.num_addon_services;

-- Q10: Does paperless billing correlate with churn?
SELECT
    a.paperless_billing,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM accounts a
JOIN churn_status c ON a.customer_id = c.customer_id
GROUP BY a.paperless_billing;

-- Q11: Does having tech support reduce churn (isolated from other factors)?
SELECT
    s.tech_support,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM services s
JOIN churn_status c ON s.customer_id = c.customer_id
GROUP BY s.tech_support;

-- Q12: Churn rate by senior citizen status.
SELECT
    cu.senior_citizen,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers cu
JOIN churn_status c ON cu.customer_id = c.customer_id
GROUP BY cu.senior_citizen;

-- Q13: Payment method breakdown — which methods have the highest churn?
SELECT
    a.payment_method,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM accounts a
JOIN churn_status c ON a.customer_id = c.customer_id
GROUP BY a.payment_method
ORDER BY churn_rate_pct DESC;
