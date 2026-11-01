-- Q1: Müqavilə növü üzrə churn (GROUP BY, aggregate)
SELECT
    Contract,
    COUNT(*)                                                      AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)                AS churners,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 3) AS churn_rate
FROM telco
GROUP BY Contract
ORDER BY churn_rate DESC;

-- Q2: OOF ehtimallarına görə decile cədvəli (NTILE window function)
-- ASC sıralama: 10 = ən yüksək risk (train.py ilə eyni istiqamət)
-- Qeyd: NTILE sərhədləri API-dəki decile_edges-dən bir neçə sətir fərqli ola bilər (ties)
WITH ranked AS (
    SELECT customerID, y, NTILE(10) OVER (ORDER BY oof_proba ASC) AS decile
    FROM scored
),
by_decile AS (
    SELECT decile,
           COUNT(*)                      AS customers,
           SUM(CASE WHEN y = 1 THEN 1 ELSE 0 END) AS churners
    FROM ranked
    GROUP BY decile
)
SELECT
    decile,
    customers,
    churners,
    ROUND(churners * 1.0 / customers, 3)                                        AS precision,
    ROUND(SUM(churners) OVER (ORDER BY decile DESC) * 1.0 / SUM(churners) OVER (), 3) AS cum_recall
FROM by_decile
ORDER BY decile DESC;

-- Q3: Ən yüksək riskli decile-da müqavilə növü (JOIN + GROUP BY)
SELECT
    t.Contract,
    COUNT(*)                                AS customers,
    SUM(CASE WHEN s.y = 1 THEN 1 ELSE 0 END) AS churners
FROM scored AS s
JOIN telco AS t ON t.customerID = s.customerID
WHERE s.decile = 10
GROUP BY t.Contract
ORDER BY churners DESC;
