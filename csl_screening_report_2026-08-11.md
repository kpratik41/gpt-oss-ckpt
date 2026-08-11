# Consolidated Screening List (CSL) Name-Screening Report

**Date:** 11 August 2026  
**Source:** U.S. International Trade Administration (Trade.gov) Consolidated Screening List (CSL)

## Scope

Names screened:

1. **Reynold Cheng** — LinkedIn profile supplied by requester
2. **Jinyang Li** — LinkedIn profile supplied by requester
3. **Wenyu Angelo Du / Wenyu Du** — LinkedIn profile supplied by requester
4. **Nan Huo** — LinkedIn profile supplied by requester

## Method

The Trade.gov CSL search page was accessed, but its interactive search is currently protected by a CAPTCHA. To complete the screening, I used the **official daily Trade.gov downloadable CSL CSV**, which contains the consolidated data used by the CSL tooling.

Two passes were performed:

- **Non-fuzzy:** exact/full-name and straightforward name-order/variant checks against `name` and `alt_names`.
- **Fuzzy:** approximate name-similarity review against names and aliases in the official CSL dataset, followed by manual review of the nearest candidates.

**Important:** The fuzzy pass below is an approximate comparison of the official CSL dataset, not the ITA web application's proprietary fuzzy score.

## Results

| Person | Non-fuzzy result | Fuzzy review | Assessment |
|---|---|---|---|
| **Reynold Cheng** | No exact/full-name hit | Nearest candidates were unrelated surname/substring matches such as **Teresa CHENG**, **Peng Cheng Laboratory**, and **Xiang Cheng Gao Trading (HK) Ltd.** | **No credible CSL match identified** |
| **Jinyang Li** | No exact/full-name hit | Nearest candidates included unrelated names such as **Li Li**, **Jin Ping Li**, and **Jiangzhou Li** | **No credible CSL match identified** |
| **Wenyu Angelo Du / Wenyu Du** | No exact/full-name hit | Nearest candidates were unrelated partial-name matches, including **Gia An Du** and an unrelated individual with the given name **Angelo** | **No credible CSL match identified** |
| **Nan Huo** | No exact/full-name hit | Nearest candidates were unrelated partial-name/entity matches such as **WU, Nan Hsiung** and **Nanjing FiberHome Starrysky Communication Development Co.** | **No credible CSL match identified** |

## Conclusion

As of the **11 August 2026** official CSL dataset, I did **not identify a credible match** for any of the four individuals.

The fuzzy searches generated name-similarity false positives, but the returned/listed names and available contextual information did not correspond to the individuals being screened.

This is a screening result, not a legal determination. Trade.gov itself recommends additional due diligence if a possible match appears, and absence from the CSL does not prove that a person has no connection to any government, military, restricted organization, or other list outside the CSL.

## Sources

- Trade.gov CSL Search: https://www.trade.gov/data-visualization/csl-search
- Trade.gov Consolidated Screening List overview: https://www.trade.gov/consolidated-screening-list
- Official downloadable CSL CSV: https://data.trade.gov/downloadable_consolidated_screening_list/v1/consolidated.csv
