"""
Correction for the first year yield reduction data.
This is neede because we start the yield reduction scenario in May.
@author: Morgan Rivers
"""


def get_year_1_ratio_using_fraction_harvest_before_may(
    first_year_xia_et_al_reduction, seasonality_values, country_iso3
):
    if country_iso3 == "ZAF":
        harvest_before_may_this_country = 1
    elif country_iso3 == "JPN":
        harvest_before_may_this_country = 0
    elif country_iso3 == "PRK":
        harvest_before_may_this_country = 0
    elif country_iso3 == "KOR":
        harvest_before_may_this_country = 0
    else:
        harvest_before_may_this_country = sum(seasonality_values[:4])

    fraction_harvest_after_may_nuclear_winter = (
        first_year_xia_et_al_reduction - harvest_before_may_this_country
    )
    if first_year_xia_et_al_reduction < 0:
        first_year_xia_et_al_reduction = 0
    assert first_year_xia_et_al_reduction < 101  # the improvement can't be that much...

    if fraction_harvest_after_may_nuclear_winter < 0:
        fraction_harvest_after_may_nuclear_winter = 0

    if fraction_harvest_after_may_nuclear_winter > 0:
        # if the expected yield for the remaining months is nonzero

        fraction_harvest_after_may = 1 - harvest_before_may_this_country

        assert fraction_harvest_after_may >= 0
        assert fraction_harvest_after_may <= 1

        if fraction_harvest_after_may < 0.25:
            # in this case, we simply don't have much data about the yield in nuclear winter from xia et al
            # year 1 data, and it doesn't make more than a 25% difference in the total yield over a 12 month
            # period, so the yield in these months is assumed to be the same.
            fraction_continued_yields = 1

        else:
            # We have significant harvest during the months May, June, July, August, September, October,
            # November, and December. The ratio of yield to expected in nuclear winter during those months
            # is calculated as the fraction_harvest_after_may_nuclear_winter divided by the
            # fraction_harvest_after_may.

            ratio_yields_nw = fraction_harvest_after_may_nuclear_winter

            fraction_continued_yields = ratio_yields_nw / fraction_harvest_after_may

    else:
        # The fraction of the normal harvest is zero after subtracting yield from normal months.
        # Therefore, the total harvest will be zero.
        fraction_continued_yields = 0

    return fraction_continued_yields


def update_yield_reduction_for_first_year(yield_reduction, monthly_seasonality):
    # Iterate through each country to compute the first-year reduction
    for country_iso3 in yield_reduction.index:
        country_data = {
            "iso3": country_iso3,
            "seasonality_m1": monthly_seasonality.at[country_iso3, "seasonality_m1"],
            "seasonality_m2": monthly_seasonality.at[country_iso3, "seasonality_m2"],
            "seasonality_m3": monthly_seasonality.at[country_iso3, "seasonality_m3"],
            "seasonality_m4": monthly_seasonality.at[country_iso3, "seasonality_m4"],
            "seasonality_m5": monthly_seasonality.at[country_iso3, "seasonality_m5"],
            "seasonality_m6": monthly_seasonality.at[country_iso3, "seasonality_m6"],
            "seasonality_m7": monthly_seasonality.at[country_iso3, "seasonality_m7"],
            "seasonality_m8": monthly_seasonality.at[country_iso3, "seasonality_m8"],
            "seasonality_m9": monthly_seasonality.at[country_iso3, "seasonality_m9"],
            "seasonality_m10": monthly_seasonality.at[country_iso3, "seasonality_m10"],
            "seasonality_m11": monthly_seasonality.at[country_iso3, "seasonality_m11"],
            "seasonality_m12": monthly_seasonality.at[country_iso3, "seasonality_m12"],
            "crop_reduction_year1": yield_reduction.at[
                country_iso3, "crop_reduction_year1"
            ],
        }
        # Calculate the first-year reduction
        first_year_adjustment = get_year_1_ratio_using_fraction_harvest_before_may(
            country_data["crop_reduction_year1"],
            [country_data[f"seasonality_m{i}"] for i in range(1, 13)],
            country_iso3,
        )

        # Adjust the total production for the first year
        yield_reduction.at[country_iso3, "crop_reduction_year1"] = first_year_adjustment
    return yield_reduction
