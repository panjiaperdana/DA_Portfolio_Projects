-- Create Table and Insert the csv file
CREATE TABLE customer_data (
	customer_id TEXT,
	gender TEXT,
	age INT,
	married TEXT,
	state TEXT,
	number_of_referrals INT,
	tenure_in_months INT,
	value_deal TEXT,
	phone_service TEXT,
	multiple_lines TEXT,
	internet_service TEXT,
	internet_type TEXT,
	online_security TEXT,
	online_backup TEXT,
	device_protection_plan TEXT,
	premium_support TEXT,
	streaming_tv TEXT,
	streaming_movies TEXT,
	streaming_music TEXT,
	unlimited_data TEXT,
	contract TEXT,
	paperless_billing TEXT,
	payment_method TEXT,
	monthly_charge NUMERIC,
	total_charges NUMERIC,
	total_refunds NUMERIC,
	total_extra_data_charges INT,
	total_long_distance_charges NUMERIC,
	total_revenue NUMERIC,
	customer_status TEXT,
	churn_category TEXT,
	churn_reason TEXT
);

COPY customer_data
FROM
'D:\Data Analyst - Tutorial\Portfolio Projects\Telecom\Source\customer_data.csv'
DELIMITER ','
CSV HEADER;


-- Checking Data
SELECT
	*
FROM
	customer_data;


-- Count Gender
SELECT
	gender,
	COUNT(*) AS total_count,
	COUNT(*) * 1.0 / (
		SELECT
			COUNT(*)
		FROM
			customer_data
	) AS total_percentage
FROM
	customer_data
GROUP BY
	gender;


-- Count Contract
SELECT
    contract,
    COUNT(*) AS total_count,
    COUNT(*) * 1.0 / (
        SELECT COUNT(*) FROM customer_data
    ) AS total_percentage
FROM customer_data
GROUP BY contract;


-- Count Customer Status
SELECT
	customer_status,
	count(*) AS total_count,
	sum(total_revenue) AS total_revenue,
	sum(total_revenue) / (
		SELECT
			sum(total_revenue)
		FROM
			customer_data
	) AS revenue_percentage
FROM
	customer_data
GROUP BY
	customer_status;


-- Count State
SELECT
	state,
	count(*) AS tolal_count,
	count(*) * 1.0 / (
		SELECT
			count(*)
		FROM
			customer_data
	) AS count_percentage
FROM
	customer_data
GROUP BY
	state
ORDER BY
	count_percentage DESC;


-- Distinct internet_type
SELECT
	DISTINCT internet_type
FROM
	customer_data;


-- Check NULL data
DROP FUNCTION null_counts(text);

CREATE OR REPLACE FUNCTION null_counts(tablename TEXT)
RETURNS TABLE(col_name TEXT, null_count BIGINT) AS $$
DECLARE
    col_rec RECORD;
    dynsql TEXT;
BEGIN
    FOR col_rec IN
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = tablename
          AND table_schema = 'public'
    LOOP
        dynsql := format(
            'SELECT %L AS col_name, COUNT(*) FILTER (WHERE %I IS NULL) AS null_count FROM %I',
            col_rec.column_name,
            col_rec.column_name,
            tablename
        );

        RETURN QUERY EXECUTE dynsql;
    END LOOP;
END $$ LANGUAGE plpgsql;

SELECT * FROM null_counts('customer_data');



-- Replace the NULL values in new tables
CREATE TABLE prod_customer_data AS
SELECT
	customer_id,
	gender,
	age,
	married,
	state,
	number_of_referrals,
	tenure_in_months,
	COALESCE(value_deal, 'None') AS value_deal,
	phone_service,
	COALESCE(multiple_lines, 'No') AS multiple_lines,
	internet_service,
	COALESCE(internet_type, 'None') AS internet_type,
	COALESCE(online_security, 'No') AS online_security,
	COALESCE(online_backup, 'No') AS online_backup,
	COALESCE(device_protection_plan, 'No') AS device_protection_plan,
	COALESCE(premium_support, 'No') AS premium_support,
	COALESCE(streaming_tv, 'No') AS streaming_tv,
	COALESCE(streaming_movies, 'No') AS streaming_movies,
	COALESCE(streaming_music, 'No') AS streaming_music,
	COALESCE(unlimited_data, 'No') AS unlimited_data,
	contract,
	paperless_billing,
	payment_method,
	monthly_charge,
	total_charges,
	total_refunds,
	total_extra_data_charges,
	total_long_distance_charges,
	total_revenue,
	customer_status,
	COALESCE(churn_category, 'Others') AS churn_category,
	COALESCE(churn_reason, 'Others') AS churn_reason
FROM
	customer_data;

SELECT
	*
FROM
	prod_customer_data;

SELECT
	*
FROM
	null_counts('prod_customer_data');


-- Create Views
CREATE VIEW vw_churn_data AS
SELECT
	*
FROM
	prod_customer_data
WHERE
	customer_status IN (
		'Churned', 'Stayed'
	);

SELECT
	*
FROM
	vw_churn_data;

	
CREATE VIEW vw_join_data AS
SELECT
	*
FROM
	prod_customer_data
WHERE
	customer_status = 'Joined';

SELECT
	*
FROM 
	vw_join_data;