# financial4all/xbrl/standardization/standard_concepts.py
"""
Standardized concept names for financial statements.

This module defines the StandardConcept enum which provides canonical concept names
with user-friendly display labels, matching edgartools' standardization approach.
"""

from enum import Enum
from typing import Optional


class StandardConcept(str, Enum):
    """
    Standardized concept names for financial statements.
    
    The enum value (string) is the display label used for presentation.
    These labels should match keys in concept mappings.
    """
    
    # Balance Sheet - Assets
    CASH_AND_EQUIVALENTS = "Cash and Cash Equivalents"
    ACCOUNTS_RECEIVABLE = "Accounts Receivable"
    INVENTORY = "Inventory"
    PREPAID_EXPENSES = "Prepaid Expenses"
    TOTAL_CURRENT_ASSETS = "Total Current Assets"
    PROPERTY_PLANT_EQUIPMENT = "Property, Plant and Equipment"
    GOODWILL = "Goodwill"
    INTANGIBLE_ASSETS = "Intangible Assets"
    TOTAL_ASSETS = "Total Assets"
    
    # Balance Sheet - Liabilities
    ACCOUNTS_PAYABLE = "Accounts Payable"
    ACCRUED_LIABILITIES = "Accrued Liabilities"
    SHORT_TERM_DEBT = "Short Term Debt"
    TOTAL_CURRENT_LIABILITIES = "Total Current Liabilities"
    LONG_TERM_DEBT = "Long Term Debt"
    DEFERRED_REVENUE = "Deferred Revenue"
    TOTAL_LIABILITIES = "Total Liabilities"
    
    # Balance Sheet - Equity
    COMMON_STOCK = "Common Stock"
    RETAINED_EARNINGS = "Retained Earnings"
    TOTAL_EQUITY = "Total Stockholders' Equity"
    
    # Income Statement - Revenue Hierarchy
    REVENUE = "Revenue"
    CONTRACT_REVENUE = "Contract Revenue"
    PRODUCT_REVENUE = "Product Revenue"
    SERVICE_REVENUE = "Service Revenue"
    SUBSCRIPTION_REVENUE = "Subscription Revenue"
    LEASING_REVENUE = "Leasing Revenue"
    
    # Industry-Specific Revenue Concepts
    AUTOMOTIVE_REVENUE = "Automotive Revenue"
    AUTOMOTIVE_LEASING_REVENUE = "Automotive Leasing Revenue"
    ENERGY_REVENUE = "Energy Revenue"
    SOFTWARE_REVENUE = "Software Revenue"
    HARDWARE_REVENUE = "Hardware Revenue"
    PLATFORM_REVENUE = "Platform Revenue"
    
    # Income Statement - Expenses
    COST_OF_REVENUE = "Cost of Revenue"
    COST_OF_GOODS_SOLD = "Cost of Goods Sold"
    COST_OF_GOODS_AND_SERVICES_SOLD = "Cost of Goods and Services Sold"
    COST_OF_SALES = "Cost of Sales"
    COSTS_AND_EXPENSES = "Costs and Expenses"
    DIRECT_OPERATING_COSTS = "Direct Operating Costs"
    GROSS_PROFIT = "Gross Profit"
    OPERATING_EXPENSES = "Operating Expenses"
    RESEARCH_AND_DEVELOPMENT = "Research and Development Expense"
    
    # Enhanced Expense Hierarchy
    SELLING_GENERAL_ADMIN = "Selling, General and Administrative Expense"
    SELLING_EXPENSE = "Selling Expense"
    GENERAL_ADMIN_EXPENSE = "General and Administrative Expense"
    MARKETING_EXPENSE = "Marketing Expense"
    SALES_EXPENSE = "Sales Expense"
    
    # Other Income Statement
    OPERATING_INCOME = "Operating Income"
    INTEREST_EXPENSE = "Interest Expense"
    INCOME_BEFORE_TAX = "Income Before Tax"
    INCOME_BEFORE_TAX_CONTINUING_OPS = "Income Before Tax from Continuing Operations"
    INCOME_TAX_EXPENSE = "Income Tax Expense"
    NET_INCOME = "Net Income"
    NET_INCOME_CONTINUING_OPS = "Net Income from Continuing Operations"
    NET_INCOME_NONCONTROLLING = "Net Income Attributable to Noncontrolling Interest"
    PROFIT_OR_LOSS = "Profit or Loss"
    
    # Cash Flow Statement
    CASH_FROM_OPERATIONS = "Net Cash from Operating Activities"
    CASH_FROM_INVESTING = "Net Cash from Investing Activities"
    CASH_FROM_FINANCING = "Net Cash from Financing Activities"
    NET_CHANGE_IN_CASH = "Net Change in Cash"
    
    # Lease-Related (Phil Oakley Framework)
    OPERATING_LEASE_PAYMENTS = "Operating Lease Payments"
    OPERATING_LEASE_LIABILITY = "Operating Lease Liability"
    OPERATING_LEASE_RIGHT_OF_USE_ASSET = "Operating Lease Right-of-Use Asset"
    FINANCE_LEASE_LIABILITY = "Finance Lease Liability"
    
    @classmethod
    def get_from_label(cls, label: str) -> Optional['StandardConcept']:
        """
        Get a StandardConcept enum by its label value.
        
        Args:
            label: The label string to look up
            
        Returns:
            The corresponding StandardConcept or None if not found
        """
        for concept in cls:
            if concept.value == label:
                return concept
        return None
    
    @classmethod
    def get_all_values(cls) -> set[str]:
        """
        Get all label values defined in the enum.
        
        Returns:
            Set of all label strings
        """
        return {concept.value for concept in cls}
