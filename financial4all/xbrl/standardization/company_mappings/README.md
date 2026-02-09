# Company-Specific Mappings

This directory contains company-specific XBRL concept mappings to handle unique taxonomy variations that differ from standard GAAP taxonomy.

## Purpose

Some companies use company-specific XBRL tags (prefixed with company identifier like `msft_`, `tsla_`, `aapl_`) or have unique reporting structures that require special handling. Company mappings allow us to:

1. **Map company-specific tags to standard concepts**: Convert company-specific XBRL tags to standardized financial concepts
2. **Handle unique revenue/expense categorizations**: Support industry-specific breakdowns (e.g., Tesla's Automotive/Energy/Service revenue)
3. **Define hierarchical relationships**: Specify how company-specific concepts relate to parent concepts
4. **Override standard mappings**: Company mappings have higher priority (priority 2) than core mappings (priority 1)

## Format

Each company mapping file follows this structure:

```json
{
  "entity_info": {
    "name": "Company Name",
    "cik": "0000000000",
    "ticker": "TICKER",
    "description": "Brief description"
  },
  "concept_mappings": {
    "Standard Concept Name": [
      "company_SpecificTag1",
      "company_SpecificTag2"
    ]
  },
  "hierarchy_rules": {
    "Parent Concept": {
      "children": ["Child Concept 1", "Child Concept 2"],
      "calculation_rule": "sum"
    }
  },
  "business_context": {
    "entity_type": "operating_company",
    "industry": "technology",
    "description": "Business context information"
  }
}
```

## Current Mappings

### MSFT (Microsoft Corporation)
- **CIK**: 0000789019
- **Mappings**: Product Revenue, Service Revenue, Subscription Revenue, Platform Revenue
- **Expenses**: Sales and Marketing Expense, Technical Support Expense
- **Hierarchy**: Revenue = Product + Service + Subscription + Platform

### TSLA (Tesla, Inc.)
- **CIK**: 1318605
- **Mappings**: Automotive Revenue, Automotive Leasing Revenue, Energy Revenue, Service Revenue
- **Hierarchy**: Revenue = Automotive + Energy + Service; Automotive includes Automotive Leasing

### BRKA (Berkshire Hathaway Inc.)
- **CIK**: 0001067983
- **Mappings**: Sales and Service Revenue, Operating Lease Revenue
- **Hierarchy**: Revenue = Sales/Service + Operating Lease
- **Context**: Holding company with diverse business operations

### AAPL (Apple Inc.)
- **CIK**: 0000320193
- **Mappings**: Interest Income, Other, net
- **Known Issues**:
  - Interest Income can be misclassified as Operating Income due to similar magnitude for cash-rich periods
  - Revenue concept changed from "Revenues" (2007-2017) to "RevenueFromContractWithCustomer" (2018+)
  - Reports "OtherIncomeExpense" without "Net" suffix (handled in standardization.py)

### NVDA (NVIDIA Corporation)
- **CIK**: 0001045810
- **Mappings**: Capital Expenditures (custom extension tags for 2013–2021)
- **CapEx tagging**: 2013–2021 NVDA combined PPE + Intangible Assets into one SCF line; in filings they may have used custom extension elements (e.g. `nvda:PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets`). Many guides state that the **Company Facts** endpoint (`https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`) returns `facts["nvda"]` for custom tags; in practice, the live API for this CIK returns only `facts["us-gaap"]`, `facts["invest"]`, `facts["srt"]`, `facts["dei"]`—no `facts["nvda"]`. The app **does** iterate all taxonomy keys and would use `facts["nvda"]` if present; for 2013–2020, `PaymentsToAcquirePropertyPlantAndEquipment` in us-gaap appears only as 10-Q (no 10-K annual), so we do not fill those years with partial data. 2022+ uses us-gaap:PaymentsToAcquireProductiveAssets.

## Priority System

Mappings are resolved with the following priority (higher priority wins):

1. **Priority 1**: Core GAAP mappings (lowest priority)
2. **Priority 2**: Company-specific mappings (higher priority)
3. **Priority 4**: Company-specific mappings when entity is detected from concept prefix (highest priority)

Example:
- `msft_ProductRevenue` with detected entity `msft` → Priority 4 → Maps to "Product Revenue"
- Standard `Revenue` tag → Priority 1 → Maps to "Revenue"

## Adding New Company Mappings

1. **Create mapping file**: Create `{ticker}_mappings.json` in this directory
2. **Follow format**: Use the structure shown above
3. **Include entity_info**: Provide company name, CIK, and ticker
4. **Define concept_mappings**: Map company-specific tags to standard concepts
5. **Add hierarchy_rules**: If needed, define parent-child relationships
6. **Document known issues**: Add any known taxonomy issues in `business_context.known_issues`

## Testing

Company mappings are automatically loaded by `MappingStore` when initialized. To verify:

```python
from financial4all.xbrl.standardization.core import MappingStore

store = MappingStore()
print(f"Loaded {len(store.company_mappings)} company mappings")

# Test a company-specific tag
result = store.get_standard_concept('msft_ProductRevenue')
print(f"msft_ProductRevenue -> {result}")  # Should return "Product Revenue"
```

## Integration

Company mappings integrate with:
- **MappingStore**: Automatically loads all `*_mappings.json` files
- **ReverseIndex**: Company mappings complement reverse index lookups
- **SynonymGroups**: Company mappings work alongside synonym groups
- **IncomeStatement/BalanceSheet/CashFlow**: Financial statement classes use mappings through MappingStore

## References

- Based on edgartools company mapping format: https://github.com/dgunning/edgartools/tree/main/edgar/xbrl/standardization/company_mappings
- Standard concepts defined in: `financial4all/xbrl/standardization/standard_concepts.py`
- Core mapping infrastructure: `financial4all/xbrl/standardization/core.py`
