# financial4all/xbrl/standardization/generate_comprehensive_mappings.py
"""
Generate comprehensive GAAP mappings from all SynonymGroups.

This script generates gaap_mappings.json in the format:
  concept_display_name -> [list of XBRL tags]

It maps all SynonymGroups to their corresponding StandardConcept display names
and includes all synonyms from all groups.
"""

import json
import os
from typing import Dict, List

from financial4all.xbrl.standardization import get_synonym_groups
from financial4all.xbrl.standardization.standard_concepts import StandardConcept


def normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison (lowercase, replace spaces/underscores/hyphens)."""
    return text.lower().replace(' ', '_').replace('-', '_')


def map_group_name_to_display_name(group_name: str) -> str:
    """
    Map a SynonymGroup name to its StandardConcept display name.
    
    Args:
        group_name: The SynonymGroup name (e.g., 'revenue', 'net_income')
        
    Returns:
        The corresponding StandardConcept display name (e.g., 'Revenue', 'Net Income')
    """
    normalized_group = normalize_for_comparison(group_name)
    
    # Try to find matching StandardConcept
    for concept in StandardConcept:
        normalized_concept = normalize_for_comparison(concept.value)
        if normalized_concept == normalized_group:
            return concept.value
    
    # Try common mappings
    common_mappings = {
        'revenue': 'Revenue',
        'cost_of_revenue': 'Cost of Revenue',
        'gross_profit': 'Gross Profit',
        'operating_expenses': 'Operating Expenses',
        'research_and_development': 'Research and Development Expense',
        'sga_expense': 'Selling, General and Administrative Expense',
        'operating_income': 'Operating Income',
        'interest_expense': 'Interest Expense',
        'interest_income': 'Interest Income',
        'other_net': 'Other, net',
        'interest_income_net': 'Interest Income (Net)',
        'income_before_tax': 'Income Before Tax',
        'income_tax_expense': 'Income Tax Expense',
        'net_income': 'Net Income',
        'earnings_per_share_basic': 'Basic EPS',
        'earnings_per_share_diluted': 'Diluted EPS',
        'depreciation_and_amortization': 'Depreciation and Amortization',
        'ebitda': 'EBITDA',
        'cash_and_equivalents': 'Cash and Cash Equivalents',
        'short_term_investments': 'Short Term Investments',
        'accounts_receivable': 'Accounts Receivable',
        'inventory': 'Inventory',
        'prepaid_expenses': 'Prepaid Expenses',
        'total_current_assets': 'Total Current Assets',
        'property_plant_equipment': 'Property, Plant and Equipment',
        'goodwill': 'Goodwill',
        'intangible_assets': 'Intangible Assets',
        'long_term_investments': 'Long Term Investments',
        'deferred_tax_assets': 'Deferred Tax Assets',
        'total_assets': 'Total Assets',
        'accounts_payable': 'Accounts Payable',
        'accrued_liabilities': 'Accrued Liabilities',
        'short_term_debt': 'Short Term Debt',
        'deferred_revenue': 'Deferred Revenue',
        'total_current_liabilities': 'Total Current Liabilities',
        'long_term_debt': 'Long Term Debt',
        'deferred_tax_liabilities': 'Deferred Tax Liabilities',
        'total_liabilities': 'Total Liabilities',
        'common_stock': 'Common Stock',
        'additional_paid_in_capital': 'Additional Paid In Capital',
        'retained_earnings': 'Retained Earnings',
        'treasury_stock': 'Treasury Stock',
        'accumulated_other_comprehensive_income': 'Accumulated Other Comprehensive Income',
        'stockholders_equity': "Total Stockholders' Equity",
        'common_shares_outstanding': 'Common Shares Outstanding',
        'operating_cash_flow': 'Net Cash from Operating Activities',
        'investing_cash_flow': 'Net Cash from Investing Activities',
        'financing_cash_flow': 'Net Cash from Financing Activities',
        'net_change_in_cash': 'Net Change in Cash',
        'capex': 'Capital Expenditures',
        'dividends_paid': 'Dividends Paid',
        'share_repurchases': 'Share Repurchases',
        'debt_repayment': 'Debt Repayment',
        'debt_proceeds': 'Debt Proceeds',
        'free_cash_flow': 'Free Cash Flow',
        'operating_lease_payments': 'Operating Lease Payments',
        'operating_lease_liability': 'Operating Lease Liability',
        'operating_lease_right_of_use_asset': 'Operating Lease Right-of-Use Asset',
        'finance_lease_liability': 'Finance Lease Liability',
        'book_value_per_share': 'Book Value Per Share',
        'return_on_equity': 'Return On Equity',
        'return_on_assets': 'Return On Assets',
    }
    
    if group_name in common_mappings:
        return common_mappings[group_name]
    
    # Fallback: convert group name to title case
    return group_name.replace('_', ' ').title()


def generate_comprehensive_gaap_mappings() -> Dict[str, List[str]]:
    """
    Generate comprehensive gaap_mappings.json from all SynonymGroups.
    
    Returns:
        Dictionary mapping standard concept display names to lists of XBRL tags
        Format: { "Revenue": ["Revenue", "Revenues", ...], ... }
    """
    synonyms = get_synonym_groups()
    mappings: Dict[str, List[str]] = {}
    
    # Iterate through all groups
    for group_name, group in synonyms._groups.items():
        # Map group name to display name
        display_name = map_group_name_to_display_name(group_name)
        
        # Initialize list if needed
        if display_name not in mappings:
            mappings[display_name] = []
        
        # Add all synonyms (they're already normalized by SynonymGroup)
        for synonym in group.synonyms:
            # Normalize synonym (remove namespace if present)
            normalized_synonym = synonym
            if ':' in normalized_synonym:
                normalized_synonym = normalized_synonym.split(':')[-1]
            if '_' in normalized_synonym:
                parts = normalized_synonym.split('_', 1)
                if len(parts) > 1 and parts[0].lower() in ('usgaap', 'dei', 'srt', 'ifrs'):
                    normalized_synonym = parts[1]
            
            # Add if not already present (avoid duplicates)
            if normalized_synonym not in mappings[display_name]:
                mappings[display_name].append(normalized_synonym)
    
    # Sort tags within each concept for consistency
    for display_name in mappings:
        mappings[display_name].sort()
    
    return mappings


def generate_comprehensive_display_names() -> Dict[str, str]:
    """
    Generate comprehensive display_names.json from all mapped concepts.
    
    Returns:
        Dictionary mapping standard concept names to display names
    """
    display_names = {}
    
    # Add all StandardConcept enum values
    for concept in StandardConcept:
        display_names[concept.value] = concept.value
    
    # Add mappings from generated GAAP mappings
    gaap_mappings = generate_comprehensive_gaap_mappings()
    for display_name in gaap_mappings:
        if display_name not in display_names:
            display_names[display_name] = display_name
    
    return display_names


def main():
    """Generate comprehensive data files."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate gaap_mappings.json
    print("Generating comprehensive gaap_mappings.json...")
    gaap_mappings = generate_comprehensive_gaap_mappings()
    gaap_path = os.path.join(module_dir, "gaap_mappings.json")
    
    # Sort by display name for consistency
    sorted_mappings = dict(sorted(gaap_mappings.items()))
    
    with open(gaap_path, 'w') as f:
        json.dump(sorted_mappings, f, indent=2)
    
    total_tags = sum(len(tags) for tags in sorted_mappings.values())
    print(f"  Generated {len(sorted_mappings)} concept mappings with {total_tags} total XBRL tags")
    
    # Generate display_names.json
    print("Generating comprehensive display_names.json...")
    display_names = generate_comprehensive_display_names()
    display_path = os.path.join(module_dir, "display_names.json")
    
    sorted_display_names = dict(sorted(display_names.items()))
    with open(display_path, 'w') as f:
        json.dump(sorted_display_names, f, indent=2)
    
    print(f"  Generated {len(sorted_display_names)} display names")
    
    print("\nAll data files generated successfully!")
    print(f"\nSummary:")
    print(f"  - Concepts mapped: {len(sorted_mappings)}")
    print(f"  - Total XBRL tags: {total_tags}")
    print(f"  - Average tags per concept: {total_tags / len(sorted_mappings):.1f}")


if __name__ == "__main__":
    main()
