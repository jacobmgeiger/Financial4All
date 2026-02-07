# Calculation-Based Validation

## Overview

The calculation-based validation system validates XBRL concept mappings by checking if they satisfy expected calculation relationships (e.g., `Revenue - COGS = Gross Profit`). This helps catch mapping errors early, such as misclassified concepts that don't satisfy expected financial relationships.

## Features

### 1. Standard Validation Rules

Pre-defined validation rules for common financial relationships:

- **Gross Profit**: `Revenue - Cost of Revenue = Gross Profit`
- **Operating Income**: `Gross Profit - Operating Expenses = Operating Income`
- **Income Before Taxes**: `Operating Income + Other income (expense), net = Income Before Taxes`
- **Net Income**: `Income Before Taxes - Taxes = Net Income`
- **Other income (expense), net**: `Interest Income + Interest Expense + Other, net = Other income (expense), net`
- **Operating Expenses**: `R&D + SG&A = Operating Expenses`

### 2. Special Validations

#### Interest Income vs Operating Income

The validator includes special logic to detect when `Interest Income` is suspiciously close to `Operating Income`, which often indicates misclassification:

- Flags if `Interest Income` is within 5% of `Operating Income`
- Flags if `Interest Income` exceeds 10% of `Operating Income`
- This helps catch cases like the AAPL Interest Income issue

### 3. Calculation Linkbase Integration

The validator can use calculation relationships from XBRL calculation linkbases to validate mappings:

- Validates parent-child relationships from calculation linkbases
- Checks if mapped concepts satisfy calculation formulas
- Uses weights from calculation arcs to validate relationships

## Usage

### Basic Usage

```python
from financial4all.xbrl.standardization import CalculationValidator
from financial4all.xbrl.calculations import CalculationEngine

# Initialize validator
validator = CalculationValidator(CalculationEngine())

# Validate a mapping
is_valid, error_msg = validator.validate_mapping(
    mapped_concept="Interest Income",
    calculation_parent="Operating Income",
    available_facts={
        "Operating Income": 100000,
        "Interest Income": 95000  # Suspiciously close!
    }
)

if not is_valid:
    print(f"Validation failed: {error_msg}")
```

### Integration with ConceptMapper

The `ConceptMapper` automatically validates mappings when `validate_with_calculations=True` (default):

```python
from financial4all.xbrl.standardization import ConceptMapper, MappingStore

mapper = ConceptMapper(MappingStore())

# Mapping is automatically validated
standard_concept = mapper.map_concept(
    company_concept="us-gaap_InterestIncome",
    label="Interest Income",
    context={
        "statement_type": "IncomeStatement",
        "available_facts": {
            "Operating Income": 100000,
            "Interest Income": 95000
        },
        "validate_with_calculations": True  # Default: True
    }
)
```

### Validating Multiple Mappings

```python
# Validate all mappings for a statement
validation_results = validator.validate_statement_mappings(
    mapped_concepts={
        "us-gaap_Revenue": "Revenue",
        "us-gaap_CostOfRevenue": "Cost of Revenue",
        "us-gaap_GrossProfit": "Gross Profit"
    },
    available_facts={
        "Revenue": 1000000,
        "Cost of Revenue": 600000,
        "Gross Profit": 400000
    }
)

for concept, (is_valid, error_msg) in validation_results.items():
    if not is_valid:
        print(f"{concept}: {error_msg}")
```

## Configuration

### Tolerance

The validator uses a tolerance for floating-point comparisons (default: 1%):

```python
validator = CalculationValidator(
    calculation_engine=CalculationEngine(),
    tolerance=0.01  # 1% tolerance (default)
)
```

### Custom Validation Rules

You can extend the validator with custom rules:

```python
validator = CalculationValidator()
validator.validation_rules["Custom Concept"] = [
    ("Child Concept 1", 1.0, "Description"),
    ("Child Concept 2", -1.0, "Description")
]
```

## Integration Points

### Income Statement Processing

The validation is integrated into the income statement processing pipeline:

1. **During Mapping**: `ConceptMapper.map_concept()` validates mappings automatically
2. **Post-Processing**: Can be used to validate final statement data
3. **Error Logging**: Validation failures are logged as warnings

### Error Handling

- Validation failures are logged but don't prevent mapping
- Allows system to continue while flagging potential issues
- Errors can be reviewed and addressed manually

## Benefits

1. **Early Error Detection**: Catches mapping errors before they propagate
2. **Automatic Validation**: No manual intervention required
3. **Comprehensive Coverage**: Validates common financial relationships
4. **Flexible**: Can be extended with custom rules
5. **Non-Blocking**: Logs warnings but doesn't prevent processing

## Example: Detecting AAPL Interest Income Issue

```python
# This would have caught the AAPL Interest Income misclassification
validator = CalculationValidator()

is_valid, error = validator.validate_mapping(
    mapped_concept="Interest Income",
    available_facts={
        "Operating Income": 133050,
        "Interest Income": 132729  # Suspiciously close!
    }
)

# Output:
# Validation failed: Interest Income (132,729) is suspiciously close to 
# Operating Income (133,050) (diff: 0.24%). This may be misclassified Operating Income.
```

## Future Enhancements

1. **More Validation Rules**: Add rules for balance sheet and cash flow relationships
2. **ML-Based Validation**: Use ML to detect unusual patterns
3. **Cross-Company Validation**: Compare mappings across companies
4. **Automated Rule Learning**: Learn validation rules from data

## See Also

- [STANDARDIZATION_GAPS_ANALYSIS.md](./STANDARDIZATION_GAPS_ANALYSIS.md) - Analysis of standardization gaps
- [calculation_validation.py](./calculation_validation.py) - Implementation details
- [core.py](./core.py) - ConceptMapper integration
