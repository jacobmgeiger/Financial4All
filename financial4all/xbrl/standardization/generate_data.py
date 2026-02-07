# financial4all/xbrl/standardization/generate_data.py
"""
Data generation script for standardization mappings.

This script generates the JSON data files needed for the standardization system:
- gaap_mappings.json: Maps XBRL tags to standard concepts
- display_names.json: Maps standard concepts to display names
- section_membership.json: Maps concepts to statement sections

Run this script to generate the data files from existing SynonymGroups.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

# Import existing standardization to get SynonymGroups
import sys
import importlib.util
import os

# Import from parent standardization.py module directly
parent_file = os.path.join(os.path.dirname(__file__), '..', 'standardization.py')
parent_spec = importlib.util.spec_from_file_location("standardization_parent", parent_file)
parent_module = importlib.util.module_from_spec(parent_spec)
parent_spec.loader.exec_module(parent_module)

# Get functions from parent module
get_synonym_groups = parent_module.get_synonym_groups
SynonymGroups = parent_module.SynonymGroups

# Import StandardConcept enum
from financial4all.xbrl.standardization.standard_concepts import StandardConcept


def generate_gaap_mappings() -> Dict[str, List[str]]:
    """
    Generate gaap_mappings.json from SynonymGroups.
    
    Returns:
        Dictionary mapping XBRL tags to lists of standard concept names
    """
    synonyms = get_synonym_groups()
    mappings = {}
    
    # Iterate through all groups
    for group_name, group in synonyms._groups.items():
        # Map group name to StandardConcept if possible
        # Try to find matching StandardConcept by label
        standard_concept = None
        for concept in StandardConcept:
            # Normalize both for comparison
            concept_label_lower = concept.value.lower().replace(' ', '_').replace('-', '_')
            group_name_lower = group_name.lower()
            
            if concept_label_lower == group_name_lower:
                standard_concept = concept.value
                break
        
        # If no exact match, use group name as standard concept
        if not standard_concept:
            standard_concept = group_name.replace('_', ' ').title()
        
        # Map each synonym to the standard concept
        for synonym in group.synonyms:
            # Normalize synonym (remove namespace if present)
            normalized_synonym = synonym
            if ':' in normalized_synonym:
                normalized_synonym = normalized_synonym.split(':')[-1]
            if '_' in normalized_synonym:
                parts = normalized_synonym.split('_', 1)
                if len(parts) > 1 and parts[0].lower() in ('usgaap', 'dei', 'srt', 'ifrs'):
                    normalized_synonym = parts[1]
            
            if normalized_synonym not in mappings:
                mappings[normalized_synonym] = []
            
            if standard_concept not in mappings[normalized_synonym]:
                mappings[normalized_synonym].append(standard_concept)
    
    return mappings


def generate_display_names() -> Dict[str, str]:
    """
    Generate display_names.json from StandardConcept enum.
    
    Returns:
        Dictionary mapping standard concept names to display labels
    """
    display_names = {}
    
    # Map StandardConcept enum values to their display labels
    for concept in StandardConcept:
        # Use the enum value (which is the display label) as both key and value
        # For now, use the display label as the key
        display_names[concept.value] = concept.value
    
    # Also add common variations
    # Map common concept name formats to display names
    concept_name_mappings = {
        "CashAndEquivalents": "Cash and Cash Equivalents",
        "AccountsReceivable": "Accounts Receivable",
        "TotalCurrentAssets": "Total Current Assets",
        "PropertyPlantEquipment": "Property, Plant and Equipment",
        "TotalAssets": "Total Assets",
        "AccountsPayable": "Accounts Payable",
        "AccruedLiabilities": "Accrued Liabilities",
        "ShortTermDebt": "Short Term Debt",
        "TotalCurrentLiabilities": "Total Current Liabilities",
        "LongTermDebt": "Long Term Debt",
        "TotalLiabilities": "Total Liabilities",
        "CommonStock": "Common Stock",
        "RetainedEarnings": "Retained Earnings",
        "TotalEquity": "Total Stockholders' Equity",
        "Revenue": "Revenue",
        "CostOfRevenue": "Cost of Revenue",
        "GrossProfit": "Gross Profit",
        "OperatingExpenses": "Operating Expenses",
        "OperatingIncome": "Operating Income",
        "NetIncome": "Net Income",
    }
    
    display_names.update(concept_name_mappings)
    
    return display_names


def generate_section_membership() -> Dict[str, Dict[str, List[str]]]:
    """
    Generate section_membership.json mapping concepts to statement sections.
    
    Returns:
        Dictionary mapping statement types to sections to concept lists
    """
    # Basic structure - this would ideally be populated from actual statement analysis
    # For now, create a basic structure based on common knowledge
    
    membership = {
        "BalanceSheet": {
            "Current Assets": [
                "Cash and Cash Equivalents",
                "Accounts Receivable",
                "Inventory",
                "Prepaid Expenses",
            ],
            "Non-Current Assets": [
                "Property, Plant and Equipment",
                "Goodwill",
                "Intangible Assets",
            ],
            "Total Assets": [
                "Total Assets",
            ],
            "Current Liabilities": [
                "Accounts Payable",
                "Accrued Liabilities",
                "Short Term Debt",
            ],
            "Non-Current Liabilities": [
                "Long Term Debt",
                "Deferred Revenue",
            ],
            "Total Liabilities": [
                "Total Liabilities",
            ],
            "Equity": [
                "Common Stock",
                "Retained Earnings",
                "Total Stockholders' Equity",
            ],
        },
        "IncomeStatement": {
            "Revenue": [
                "Revenue",
            ],
            "Costs": [
                "Cost of Revenue",
                "Cost of Goods Sold",
            ],
            "Operating Expenses": [
                "Operating Expenses",
                "Research and Development Expense",
                "Selling, General and Administrative Expense",
            ],
            "Income": [
                "Gross Profit",
                "Operating Income",
                "Net Income",
            ],
        },
        "CashFlowStatement": {
            "Operating Activities": [
                "Net Cash from Operating Activities",
            ],
            "Investing Activities": [
                "Net Cash from Investing Activities",
            ],
            "Financing Activities": [
                "Net Cash from Financing Activities",
            ],
            "Net Change": [
                "Net Change in Cash",
            ],
        },
    }
    
    return membership


def main():
    """Generate all data files."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate gaap_mappings.json
    print("Generating gaap_mappings.json...")
    gaap_mappings = generate_gaap_mappings()
    gaap_path = os.path.join(module_dir, "gaap_mappings.json")
    with open(gaap_path, 'w') as f:
        json.dump(gaap_mappings, f, indent=2)
    print(f"  Generated {len(gaap_mappings)} mappings")
    
    # Generate display_names.json
    print("Generating display_names.json...")
    display_names = generate_display_names()
    display_path = os.path.join(module_dir, "display_names.json")
    with open(display_path, 'w') as f:
        json.dump(display_names, f, indent=2)
    print(f"  Generated {len(display_names)} display names")
    
    # Generate section_membership.json
    print("Generating section_membership.json...")
    section_membership = generate_section_membership()
    section_path = os.path.join(module_dir, "section_membership.json")
    with open(section_path, 'w') as f:
        json.dump(section_membership, f, indent=2)
    
    total_sections = sum(len(sections) for sections in section_membership.values())
    total_concepts = sum(len(concepts) for sections in section_membership.values() for concepts in sections.values())
    print(f"  Generated {total_sections} sections with {total_concepts} concepts")
    
    print("\nAll data files generated successfully!")


if __name__ == "__main__":
    main()
