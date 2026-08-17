select
    timestamp,
    asset_type,
    sum(power_mw) as total_power_mw,
    avg(market_price_eur_mwh) as avg_market_price_eur_mwh,
    avg(temperature_c) as avg_temperature_c,
    avg(wind_speed_ms) as avg_wind_speed_ms,
    avg(cloud_cover) as avg_cloud_cover,
    avg(grid_imbalance) as avg_grid_imbalance
from {{ ref('stg_energy_observations') }}
group by 1,2
