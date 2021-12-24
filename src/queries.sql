
drop table assets;

SELECT create_hypertable('assets', 'date');

SELECT extract(epoch from date) as last_date
FROM assets
ORDER BY date DESC 
LIMIT 1;

select exists(select 1 from assets);