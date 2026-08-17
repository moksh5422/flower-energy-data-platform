select
    cast(timestamp as timestamp) as timestamp,
    asset_id, asset_type, temperature_c, wind_speed_ms, cloud_cover,
    market_price_eur_mwh, grid_imbalance, power_mw, state_of_charge
from {{ source('raw','energy_observations') }}
where timestamp is not null
