# financial4all/xbrl/standardization.py
"""
Unified Standardization Infrastructure for Financial4All.

This module provides centralized standardization components for XBRL concepts,
enabling consistent cross-company financial analysis. Inspired by EdgarTools'
standardization approach.

Components:
 - SynonymGroups: Unified synonym management for XBRL tags
 - ConceptInfo: Rich metadata about identified concepts
 - StandardizationStore: Legacy compatibility layer

Example:
 >>> from financial4all.xbrl.standardization import get_synonym_groups
 >>>
 >>> # Get default singleton instance
 >>> synonyms = get_synonym_groups()
 >>>
 >>> # Look up synonyms for a concept
 >>> tags = synonyms.get_synonyms('revenue')
 >>> print(tags[:2])
 ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues']
 >>>
 >>> # Identify what concept a tag represents
 >>> info = synonyms.identify_concept('us-gaap:Revenues')
 >>> print(info.name)
 'revenue'
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from financial4all.core import log

# Module-level caches
_default_instance: Optional['SynonymGroups'] = None
_builtin_groups_cache: Optional[List['SynonymGroup']] = None


def _normalize_name(name: str) -> str:
    """Normalize a concept name to lowercase with underscores."""
    return name.strip().lower().replace(' ', '_').replace('-', '_')


@dataclass
class SynonymGroup:
    """
    A group of XBRL tags that represent the same financial concept.
    
    Attributes:
        name: Canonical name for the concept (e.g., 'revenue', 'net_income')
        synonyms: List of XBRL tag names that represent this concept
        description: Human-readable description of the concept
        namespace: Default namespace for tags (default: 'us-gaap')
        priority_order: How to order synonyms when resolving
            - 'listed': Use order as specified in synonyms list
            - 'frequency': Order by usage frequency (most common first)
            - 'specificity': Order by tag specificity (most specific first)
        category: Financial statement category (e.g., 'income_statement', 'balance_sheet')
    """
    name: str
    synonyms: List[str]
    description: str = ""
    namespace: str = "us-gaap"
    priority_order: str = "listed"
    category: str = ""
    # Internal set for O(1) tag membership lookup (not serialized)
    _synonym_set: Set[str] = field(default_factory=set, repr=False, compare=False)
    
    def __post_init__(self):
        """Normalize the synonym group after initialization."""
        # Ensure name is lowercase with underscores
        self.name = _normalize_name(self.name)
        # Remove namespace prefixes and deduplicate while preserving order
        seen: Set[str] = set()
        deduped: List[str] = []
        for s in self.synonyms:
            stripped = self._strip_namespace(s)
            key = stripped.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(stripped)
        self.synonyms = deduped
        # Reuse the set we already built for O(1) lookup
        self._synonym_set = seen
    
    @staticmethod
    def _strip_namespace(tag: str) -> str:
        """Remove namespace prefix from tag (e.g., 'us-gaap:Revenue' -> 'Revenue')."""
        if ':' in tag:
            return tag.split(':', 1)[1]
        # Handle underscore format (us-gaap_Revenue)
        if '_' in tag:
            parts = tag.split('_', 1)
            if parts[0].replace('-', '') in ('usgaap', 'dei', 'srt', 'ifrs'):
                return parts[1]
        return tag
    
    def get_tags_with_namespace(self, namespace: Optional[str] = None) -> List[str]:
        """
        Get synonyms with namespace prefix.
        
        Args:
            namespace: Namespace to use (default: self.namespace)
            
        Returns:
            List of tags with namespace prefix
        """
        ns = namespace or self.namespace
        return [f"{ns}:{tag}" for tag in self.synonyms]
    
    def contains_tag(self, tag: str) -> bool:
        """
        Check if this group contains the given tag.
        
        Args:
            tag: XBRL tag to check (with or without namespace)
            
        Returns:
            True if tag is in this group's synonyms
        """
        normalized = self._strip_namespace(tag).lower()
        return normalized in self._synonym_set  # O(1) lookup
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'synonyms': self.synonyms,
            'description': self.description,
            'namespace': self.namespace,
            'priority_order': self.priority_order,
            'category': self.category
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SynonymGroup':
        """Create SynonymGroup from dictionary."""
        return cls(
            name=data['name'],
            synonyms=data['synonyms'],
            description=data.get('description', ''),
            namespace=data.get('namespace', 'us-gaap'),
            priority_order=data.get('priority_order', 'listed'),
            category=data.get('category', '')
        )


@dataclass
class ConceptInfo:
    """
    Information about an identified concept from a tag lookup.
    
    Attributes:
        name: Canonical concept name
        tag: The original tag that was looked up
        group: The full SynonymGroup containing this concept
        match_type: How the match was found ('exact', 'normalized', 'fuzzy')
    """
    name: str
    tag: str
    group: SynonymGroup
    match_type: str = "exact"
    
    @property
    def synonyms(self) -> List[str]:
        """Get all synonyms for this concept."""
        return self.group.synonyms
    
    @property
    def description(self) -> str:
        """Get concept description."""
        return self.group.description
    
    @property
    def category(self) -> str:
        """Get concept category."""
        return self.group.category


def _get_builtin_groups_cached() -> List[SynonymGroup]:
    """
    Get the pre-built synonym groups (cached at module level).
    
    These groups are based on:
    - Existing STANDARD_MAPPING dictionaries from financial statement classes
    - Common XBRL concept variations
    - Industry-standard financial reporting practices
    
    The synonyms are ordered by priority (most common/specific first).
    """
    global _builtin_groups_cache
    if _builtin_groups_cache is not None:
        return _builtin_groups_cache
    
    _builtin_groups_cache = [
        # ═══════════════════════════════════════════════════════════════════
        # INCOME STATEMENT CONCEPTS
        # ═══════════════════════════════════════════════════════════════════
        SynonymGroup(
            name='revenue',
            synonyms=[
                'RevenueFromContractWithCustomerExcludingAssessedTax',
                'RevenueFromContractWithCustomerIncludingAssessedTax',
                'Revenues',
                'Revenue',
                'SalesRevenueNet',
                'SalesRevenueGoodsNet',
                'SalesRevenueNetOfReturnsAndAllowances',
                'TotalRevenues',
                'NetSales',
                'OperatingRevenue',
                'RevenuesNetOfInterestExpense',
            ],
            description='Total revenue/sales from operations',
            category='income_statement'
        ),
        SynonymGroup(
            name='cost_of_revenue',
            synonyms=[
                'CostOfRevenue',
                'CostOfGoodsAndServicesSold',
                'CostOfGoodsSold',
                'CostOfSales',
                'CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization',
                'DirectOperatingCosts',
            ],
            description='Cost of revenue/goods sold',
            category='income_statement'
        ),
        SynonymGroup(
            name='gross_profit',
            synonyms=[
                'GrossProfit',
            ],
            description='Revenue minus cost of revenue',
            category='income_statement'
        ),
        SynonymGroup(
            name='operating_expenses',
            synonyms=[
                'OperatingExpenses',
                'OperatingCostsAndExpenses',
                'NoninterestExpense',
                'CostsAndExpenses',
            ],
            description='Total operating expenses',
            category='income_statement'
        ),
        SynonymGroup(
            name='research_and_development',
            synonyms=[
                'ResearchAndDevelopmentExpense',
                'ResearchAndDevelopment',
                'ResearchAndDevelopmentCosts',
            ],
            description='Research and development expenses',
            category='income_statement'
        ),
        SynonymGroup(
            name='sga_expense',
            synonyms=[
                'SellingGeneralAndAdministrativeExpense',
                'GeneralAndAdministrativeExpense',
                'SellingAndMarketingExpense',
                'SellingExpense',
                'AdministrativeExpense',
            ],
            description='Selling, general and administrative expenses',
            category='income_statement'
        ),
        SynonymGroup(
            name='operating_income',
            synonyms=[
                'OperatingIncomeLoss',
                'OperatingIncome',
                'IncomeFromOperations',
                'IncomeLossFromContinuingOperationsBeforeInterestAndTaxes',
            ],
            description='Operating income/loss',
            category='income_statement'
        ),
        SynonymGroup(
            name='interest_expense',
            synonyms=[
                'InterestExpense',
                'InterestAndDebtExpense',
                'InterestExpenseOperating',
                'InterestExpenseNonoperating',
                'InterestAndFeeExpense',
            ],
            description='Interest expense',
            category='income_statement'
        ),
        SynonymGroup(
            name='interest_income',
            synonyms=[
                'InterestIncome',
                'InterestIncomeOperating',
                'InterestIncomeNonoperating',
                'InterestAndDividendIncome',
                'InterestAndDividendIncomeSecurities',
                'InterestAndFeeIncomeLoansAndLeases',
                'InterestIncomeDebtSecuritiesOperating',
                'InterestIncomeDepositsWithFinancialInstitutions',
                'InterestIncomeFederalFundsSoldAndSecuritiesPurchasedUnderAgreementsToResell',
                'InterestIncomePurchasedReceivables',
                'InvestmentIncomeInterest',
                'OtherInterestAndDividendIncome',
            ],
            description='Interest income',
            category='income_statement'
        ),
        SynonymGroup(
            name='other_net',
            synonyms=[
                'OtherIncomeExpenseNet',
                'OtherIncomeExpense',  # AAPL reports this without "Net"
                'OtherNonoperatingIncomeExpense',
                'OtherOperatingIncomeExpenseNet',
            ],
            description='Other income/expense, net',
            category='income_statement'
        ),
        SynonymGroup(
            name='interest_income_net',
            synonyms=[
                'InterestIncomeExpenseNet',
                'InterestIncomeExpenseAfterProvisionForLoanLoss',
                'NetInvestmentIncome',
            ],
            description='Net interest income/expense (Income - Expense)',
            category='income_statement'
        ),
        SynonymGroup(
            name='income_before_tax',
            synonyms=[
                'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
                'IncomeLossFromContinuingOperationsBeforeIncomeTaxes',
                'IncomeLossBeforeIncomeTaxes',
            ],
            description='Income before income taxes',
            category='income_statement'
        ),
        SynonymGroup(
            name='income_tax_expense',
            synonyms=[
                'IncomeTaxExpenseBenefit',
                'ProvisionForIncomeTaxes',
                'IncomeTaxesPaidNet',
            ],
            description='Income tax expense/benefit',
            category='income_statement'
        ),
        SynonymGroup(
            name='net_income',
            synonyms=[
                'NetIncomeLoss',
                'ProfitLoss',
                'NetIncome',
                'NetEarnings',
                'NetIncomeLossAttributableToParent',
                'NetIncomeLossAvailableToCommonStockholdersBasic',
                'NetIncomeLossAvailableToCommonStockholdersDiluted',
                'IncomeLossFromContinuingOperations',
            ],
            description='Net income/loss',
            category='income_statement'
        ),
        SynonymGroup(
            name='earnings_per_share_basic',
            synonyms=[
                'EarningsPerShareBasic',
                'EarningsPerShare',
            ],
            description='Basic earnings per share',
            category='income_statement'
        ),
        SynonymGroup(
            name='earnings_per_share_diluted',
            synonyms=[
                'EarningsPerShareDiluted',
                'EarningsPerShareDilutedOne',
            ],
            description='Diluted earnings per share',
            category='income_statement'
        ),
        SynonymGroup(
            name='weighted_average_shares_outstanding_basic',
            synonyms=[
                'WeightedAverageNumberOfSharesOutstandingBasic',
                'WeightedAverageNumberOfSharesOutstanding',
            ],
            description='Weighted average number of shares outstanding, basic',
            category='income_statement'
        ),
        SynonymGroup(
            name='weighted_average_shares_outstanding_diluted',
            synonyms=[
                'WeightedAverageNumberOfDilutedSharesOutstanding',
                'WeightedAverageNumberDilutedSharesOutstanding',
            ],
            description='Weighted average number of shares outstanding, diluted',
            category='income_statement'
        ),
        SynonymGroup(
            name='depreciation_and_amortization',
            synonyms=[
                'DepreciationAndAmortization',
                'DepreciationDepletionAndAmortization',
                'Depreciation',
                'AmortizationOfIntangibleAssets',
                'DepreciationAmortizationAndAccretionNet',
                'DepreciationAndAmortizationExpense',
                'DepreciationAmortizationAndImpairment',
                'DepreciationAmortizationAndAccretion',
                'DepreciationAmortizationAndDepletion',
                'DepreciationAndAmortizationNoncash',
                'DepreciationAmortizationAndAccretionNetOfAmortizationOfDeferredFinancingCosts',
                'CostDepreciationAmortizationAndDepletion',
                'DepreciationAmortizationAndAccretionExpense',
                'DepreciationAndAmortization',
                'DepreciationAmortizationAndDepletionExpense',
                'DepreciationAmortizationDepletionAndImpairment',
            ],
            description='Depreciation and amortization expense',
            category='income_statement'
        ),
        SynonymGroup(
            name='ebitda',
            synonyms=[
                'EBITDA',
                'EarningsBeforeInterestTaxesDepreciationAndAmortization',
            ],
            description='Earnings before interest, taxes, depreciation and amortization',
            category='income_statement'
        ),
        
        # ═══════════════════════════════════════════════════════════════════
        # BALANCE SHEET - ASSETS
        # ═══════════════════════════════════════════════════════════════════
        SynonymGroup(
            name='cash_and_equivalents',
            synonyms=[
                'CashAndCashEquivalentsAtCarryingValue',
                'CashCashEquivalentsAndShortTermInvestments',
                'CashEquivalentsAtCarryingValue',
                'Cash',
            ],
            description='Cash and cash equivalents',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='short_term_investments',
            synonyms=[
                'ShortTermInvestments',
                'MarketableSecuritiesCurrent',
                'AvailableForSaleSecuritiesDebtSecuritiesCurrent',
            ],
            description='Short-term investments and marketable securities',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='accounts_receivable',
            synonyms=[
                'AccountsReceivableNetCurrent',
                'AccountsReceivableNet',
                'ReceivablesNetCurrent',
                'AccountsReceivableGross',
            ],
            description='Accounts receivable',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='inventory',
            synonyms=[
                'InventoryNet',
                'InventoryGross',
                'InventoryFinishedGoods',
            ],
            description='Inventory',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='prepaid_expenses',
            synonyms=[
                'PrepaidExpenseAndOtherAssetsCurrent',
                'PrepaidExpenseCurrent',
                'PrepaidExpense',
            ],
            description='Prepaid expenses',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='total_current_assets',
            synonyms=[
                'AssetsCurrent',
                'CurrentAssets',
            ],
            description='Total current assets',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='property_plant_equipment',
            synonyms=[
                'PropertyPlantAndEquipmentNet',
                'PropertyPlantAndEquipmentGross',
                'FixedAssets',
            ],
            description='Property, plant and equipment',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='goodwill',
            synonyms=[
                'Goodwill',
            ],
            description='Goodwill',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='intangible_assets',
            synonyms=[
                'IntangibleAssetsNetExcludingGoodwill',
                'IntangibleAssetsNetIncludingGoodwill',
                'FiniteLivedIntangibleAssetsNet',
            ],
            description='Intangible assets',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='long_term_investments',
            synonyms=[
                'LongTermInvestments',
                'MarketableSecuritiesNoncurrent',
                'AvailableForSaleSecuritiesDebtSecuritiesNoncurrent',
            ],
            description='Long-term investments',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='deferred_tax_assets',
            synonyms=[
                'DeferredIncomeTaxAssetsNet',
                'DeferredTaxAssetsNet',
                'DeferredTaxAssetsNetCurrent',
                'DeferredTaxAssetsNetNoncurrent',
            ],
            description='Deferred tax assets',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='total_assets',
            synonyms=[
                'Assets',
                'AssetsTotal',
            ],
            description='Total assets',
            category='balance_sheet'
        ),
        
        # ═══════════════════════════════════════════════════════════════════
        # BALANCE SHEET - LIABILITIES
        # ═══════════════════════════════════════════════════════════════════
        SynonymGroup(
            name='accounts_payable',
            synonyms=[
                'AccountsPayableCurrent',
                'AccountsPayableTradeCurrent',
                'AccountsPayable',
            ],
            description='Accounts payable',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='accrued_liabilities',
            synonyms=[
                'AccruedLiabilitiesCurrent',
                'OtherAccruedLiabilitiesCurrent',
                'EmployeeRelatedLiabilitiesCurrent',
            ],
            description='Accrued liabilities',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='short_term_debt',
            synonyms=[
                'DebtCurrent',
                'ShortTermBorrowings',
                'LongTermDebtCurrent',
                'NotesPayableCurrent',
            ],
            description='Short-term debt',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='deferred_revenue',
            synonyms=[
                'DeferredRevenue',
                'DeferredRevenueCurrent',
                'DeferredRevenueNoncurrent',
                'ContractWithCustomerLiability',
                'ContractWithCustomerLiabilityCurrent',
            ],
            description='Deferred revenue / contract liabilities',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='total_current_liabilities',
            synonyms=[
                'LiabilitiesCurrent',
                'CurrentLiabilities',
            ],
            description='Total current liabilities',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='long_term_debt',
            synonyms=[
                'LongTermDebtNoncurrent',
                'LongTermDebt',
                'LongTermDebtAndCapitalLeaseObligations',
                'LongTermBorrowings',
                'LongTermNotesAndLoans',
            ],
            description='Long-term debt',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='deferred_tax_liabilities',
            synonyms=[
                'DeferredIncomeTaxLiabilitiesNet',
                'DeferredTaxLiabilitiesNoncurrent',
            ],
            description='Deferred tax liabilities',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='total_liabilities',
            synonyms=[
                'Liabilities',
                'LiabilitiesTotal',
            ],
            description='Total liabilities',
            category='balance_sheet'
        ),
        
        # ═══════════════════════════════════════════════════════════════════
        # BALANCE SHEET - EQUITY
        # ═══════════════════════════════════════════════════════════════════
        SynonymGroup(
            name='common_stock',
            synonyms=[
                'CommonStockValue',
                'CommonStocksIncludingAdditionalPaidInCapital',
                'StockholdersEquityCommonStock',
            ],
            description='Common stock value',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='additional_paid_in_capital',
            synonyms=[
                'AdditionalPaidInCapital',
                'AdditionalPaidInCapitalCommonStock',
            ],
            description='Additional paid-in capital',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='retained_earnings',
            synonyms=[
                'RetainedEarningsAccumulatedDeficit',
                'RetainedEarnings',
            ],
            description='Retained earnings',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='treasury_stock',
            synonyms=[
                'TreasuryStockValue',
                'TreasuryStockCommonValue',
            ],
            description='Treasury stock',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='accumulated_other_comprehensive_income',
            synonyms=[
                'AccumulatedOtherComprehensiveIncomeLossNetOfTax',
                'AccumulatedOtherComprehensiveIncomeLoss',
            ],
            description='Accumulated other comprehensive income/loss',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='stockholders_equity',
            synonyms=[
                'StockholdersEquity',
                'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                'StockholdersEquityAttributableToParent',
                'EquityAttributableToParent',
                'Equity',
                'ShareholdersEquity',
                'TotalEquity',
                'PartnersCapital',
                'MembersEquity',
            ],
            description='Total stockholders equity',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='common_shares_outstanding',
            synonyms=[
                'CommonStockSharesOutstanding',
                'WeightedAverageNumberOfSharesOutstandingBasic',
            ],
            description='Common shares outstanding',
            category='balance_sheet'
        ),
        
        # ═══════════════════════════════════════════════════════════════════
        # CASH FLOW STATEMENT
        # ═══════════════════════════════════════════════════════════════════
        SynonymGroup(
            name='operating_cash_flow',
            synonyms=[
                'NetCashProvidedByUsedInOperatingActivities',
                'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
                'CashFlowFromOperatingActivities',
            ],
            description='Net cash from operating activities',
            category='cash_flow'
        ),
        SynonymGroup(
            name='investing_cash_flow',
            synonyms=[
                'NetCashProvidedByUsedInInvestingActivities',
                'NetCashProvidedByUsedInInvestingActivitiesContinuingOperations',
                'CashFlowFromInvestingActivities',
            ],
            description='Net cash from investing activities',
            category='cash_flow'
        ),
        SynonymGroup(
            name='financing_cash_flow',
            synonyms=[
                'NetCashProvidedByUsedInFinancingActivities',
                'NetCashProvidedByUsedInFinancingActivitiesContinuingOperations',
                'CashFlowFromFinancingActivities',
            ],
            description='Net cash from financing activities',
            category='cash_flow'
        ),
        SynonymGroup(
            name='net_change_in_cash',
            synonyms=[
                'CashAndCashEquivalentsPeriodIncreaseDecrease',
                'IncreaseDecreaseInCashAndCashEquivalents',
            ],
            description='Net change in cash and cash equivalents',
            category='cash_flow'
        ),
        SynonymGroup(
            name='capex',
            synonyms=[
                # Tier 1: Comprehensive tags (aggregate multiple categories) - HIGHEST PRIORITY
                # These tags include both PP&E and intangible assets
                'PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets',
                'PaymentsToAcquirePropertyPlantAndEquipmentAndOtherAssets',
                'PaymentsForPropertyPlantAndEquipmentAndIntangibleAssets',
                'PaymentsForAcquisitionOfPropertyPlantAndEquipmentAndIntangibleAssets',
                'InvestmentsInPropertyPlantAndEquipmentAndIntangibleAssets',
                'PaymentsForPropertyPlantAndEquipmentAndOtherAssets',
                'CapitalExpendituresIncludingIntangibleAssets',
                
                # Tier 2: General comprehensive tags (may include multiple categories)
                'CapitalExpenditures',
                'CapitalExpendituresNet',
                'PaymentsForCapitalExpenditures',
                # Note: CapitalExpendituresIncurredButNotYetPaid and CapitalExpendituresDiscontinuedOperations
                # are excluded - they're not actual cash outflows
                
                # Tier 3: PP&E-specific comprehensive tags (edgartools-aligned)
                'PaymentsToAcquirePropertyPlantAndEquipment',
                'PaymentsForPropertyPlantAndEquipment',
                'PaymentsForPurchaseOfPropertyPlantAndEquipment',
                'InvestmentsInPropertyPlantAndEquipment',
                'CapitalExpendituresForPropertyPlantAndEquipment',
                'PaymentsForAcquisitionOfPropertyPlantAndEquipment',
                'PaymentsToAcquirePropertyPlantAndEquipmentNet',
                'PurchaseOfPropertyPlantAndEquipment',
                'PaymentsToAcquireProductiveAssets',
                'PaymentsToAcquireOtherProductiveAssets',
                'PaymentsToAcquireOtherPropertyPlantAndEquipment',
                'PaymentsToAcquireAssets',
                
                # Tier 4: Intangible assets and software (component tags)
                'PaymentsToAcquireIntangibleAssets',
                'PaymentsForSoftwareAndWebSiteDevelopmentCosts',
                'PaymentsForDevelopmentOfRealEstate',
                'CapitalExpendituresIncludingSoftware',
                'PaymentsForIntangibleAssets',
                'PaymentsToAcquireSoftware',
                
                # Tier 5: PP&E component tags (common in older filings; aggregate when no comprehensive tag)
                'PaymentsToAcquireMachineryAndEquipment',
                'PaymentsToAcquireBuildings',
                'PaymentsToAcquireFurnitureAndFixtures',
                'PaymentsToAcquireAndDevelopRealEstate',
                'PaymentsToAcquireRealEstate',
                'PaymentsToAcquireLand',
                'PaymentsToAcquireLandHeldForUse',
                'PaymentsToAcquireMiningAssets',
                'PaymentsToAcquireOilAndGasPropertyAndEquipment',
                'PaymentsToAcquireOilAndGasEquipment',
                'PaymentsToAcquireOilAndGasProperty',
                'PaymentsToAcquireTimberlands',
                'PaymentsToAcquireWaterAndWasteWaterSystems',
                'PaymentsToAcquireWaterSystems',
                'PaymentsToAcquireWasteWaterSystems',
                'PaymentsToAcquireMineralRights',
                'PaymentsToAcquireEquipmentOnLease',
                
                # Note: Business acquisition tags are EXCLUDED from CapEx
                # They are handled separately and should not be aggregated with CapEx
                # Tags like 'PaymentsToAcquireBusinessesNetOfCashAcquired' are intentionally omitted
            ],
            description='Capital expenditures (excludes business acquisitions)',
            category='cash_flow'
        ),
        SynonymGroup(
            name='dividends_paid',
            synonyms=[
                'PaymentsOfDividends',
                'PaymentsOfDividendsCommonStock',
                'DividendsPaid',
            ],
            description='Dividends paid',
            category='cash_flow'
        ),
        SynonymGroup(
            name='share_repurchases',
            synonyms=[
                'PaymentsForRepurchaseOfCommonStock',
                'StockRepurchasedDuringPeriodValue',
                'PaymentsForRepurchaseOfEquity',
            ],
            description='Share repurchases/buybacks',
            category='cash_flow'
        ),
        SynonymGroup(
            name='debt_repayment',
            synonyms=[
                'RepaymentsOfLongTermDebt',
                'RepaymentsOfDebt',
                'RepaymentsOfShortTermDebt',
            ],
            description='Debt repayments',
            category='cash_flow'
        ),
        SynonymGroup(
            name='debt_proceeds',
            synonyms=[
                'ProceedsFromIssuanceOfLongTermDebt',
                'ProceedsFromDebtNetOfIssuanceCosts',
                'ProceedsFromIssuanceOfDebt',
            ],
            description='Proceeds from debt issuance',
            category='cash_flow'
        ),
        SynonymGroup(
            name='free_cash_flow',
            synonyms=[
                'FreeCashFlow',
            ],
            description='Free cash flow (operating cash flow minus capex)',
            category='cash_flow'
        ),
        
        # ═══════════════════════════════════════════════════════════════════
        # OPERATING ACTIVITIES DETAIL
        # ═══════════════════════════════════════════════════════════════════
        SynonymGroup(
            name='stock_based_compensation',
            synonyms=[
                'ShareBasedCompensation',
                'StockBasedCompensationExpense',
                'ShareBasedCompensationExpense',
                'StockBasedCompensation',
                'ShareBasedCompensationRequisiteServicePeriodRecognitionValue',
            ],
            description='Stock-based compensation expense (non-cash)',
            category='cash_flow'
        ),
        SynonymGroup(
            name='change_in_accounts_receivable',
            synonyms=[
                'IncreaseDecreaseInAccountsReceivable',
                'IncreaseDecreaseInAccountsReceivableAndUnbilledReceivables',
                'IncreaseDecreaseInReceivables',
            ],
            description='Change in accounts receivable (operating activities)',
            category='cash_flow'
        ),
        SynonymGroup(
            name='change_in_inventory',
            synonyms=[
                'IncreaseDecreaseInInventories',
                'IncreaseDecreaseInInventory',
            ],
            description='Change in inventory (operating activities)',
            category='cash_flow'
        ),
        SynonymGroup(
            name='change_in_accounts_payable',
            synonyms=[
                'IncreaseDecreaseInAccountsPayable',
                'IncreaseDecreaseInAccountsPayableAndAccruedLiabilities',
                'IncreaseDecreaseInPayables',
            ],
            description='Change in accounts payable (operating activities)',
            category='cash_flow'
        ),
        SynonymGroup(
            name='deferred_tax_expense',
            synonyms=[
                'DeferredIncomeTaxExpenseBenefit',
                'DeferredTaxExpenseBenefit',
                'ProvisionForDeferredIncomeTaxes',
            ],
            description='Deferred tax expense/benefit (non-cash)',
            category='cash_flow'
        ),
        SynonymGroup(
            name='gain_loss_on_asset_disposal',
            synonyms=[
                'GainLossOnSaleOfAssets',
                'GainLossOnDisposalOfAssets',
                'GainLossOnSaleOfPropertyPlantAndEquipment',
            ],
            description='Gain/loss on asset disposal (non-cash)',
            category='cash_flow'
        ),
        
        # ═══════════════════════════════════════════════════════════════════
        # INVESTING ACTIVITIES DETAIL
        # ═══════════════════════════════════════════════════════════════════
        SynonymGroup(
            name='proceeds_from_sale_of_assets',
            synonyms=[
                'ProceedsFromSaleOfPropertyPlantAndEquipment',
                'ProceedsFromSaleOfAssets',
                'ProceedsFromDisposalOfPropertyPlantAndEquipment',
                'ProceedsFromSaleAndMaturityOfInvestments',
            ],
            description='Proceeds from sale of assets',
            category='cash_flow'
        ),
        SynonymGroup(
            name='business_acquisitions',
            synonyms=[
                'PaymentsToAcquireBusinessesNetOfCashAcquired',
                'PaymentsToAcquireBusinesses',
                'AcquisitionsNetOfCashAcquired',
                'PaymentsToAcquireBusinessesAndIntangibleAssets',
            ],
            description='Business acquisitions (excluded from CapEx)',
            category='cash_flow'
        ),
        SynonymGroup(
            name='purchases_of_investments',
            synonyms=[
                'PaymentsToAcquireAvailableForSaleSecurities',
                'PaymentsToAcquireDebtSecurities',
                'PaymentsToAcquireEquitySecurities',
                'PurchasesOfInvestments',
            ],
            description='Purchases of investments',
            category='cash_flow'
        ),
        SynonymGroup(
            name='proceeds_from_sale_of_investments',
            synonyms=[
                'ProceedsFromSaleOfAvailableForSaleSecurities',
                'ProceedsFromSaleOfDebtSecurities',
                'ProceedsFromSaleOfEquitySecurities',
                'ProceedsFromSaleOfInvestments',
            ],
            description='Proceeds from sale of investments',
            category='cash_flow'
        ),
        
        # ═══════════════════════════════════════════════════════════════════
        # FINANCING ACTIVITIES DETAIL
        # ═══════════════════════════════════════════════════════════════════
        SynonymGroup(
            name='proceeds_from_stock_issuance',
            synonyms=[
                'ProceedsFromIssuanceOfCommonStock',
                'ProceedsFromIssuanceOfStock',
                'ProceedsFromIssuanceOfShares',
                'ProceedsFromStockOptionsExercised',
            ],
            description='Proceeds from stock issuance',
            category='cash_flow'
        ),
        SynonymGroup(
            name='payments_for_debt_issuance_costs',
            synonyms=[
                'PaymentsOfDebtIssuanceCosts',
                'DebtIssuanceCosts',
            ],
            description='Payments for debt issuance costs',
            category='cash_flow'
        ),
        
        # ═══════════════════════════════════════════════════════════════════
        # LEASE-RELATED (Phil Oakley Framework)
        # ═══════════════════════════════════════════════════════════════════
        SynonymGroup(
            name='operating_lease_payments',
            synonyms=[
                'OperatingLeasePayments',
                'PaymentsForOperatingLeases',
                'LesseeOperatingLeaseLiabilityPaymentsDue',
                'OperatingLeasesFutureMinimumPaymentsDue',
            ],
            description='Operating lease payments (Phil Oakley framework)',
            category='cash_flow'
        ),
        SynonymGroup(
            name='operating_lease_liability',
            synonyms=[
                'OperatingLeaseLiability',
                'OperatingLeaseLiabilityCurrent',
                'OperatingLeaseLiabilityNoncurrent',
            ],
            description='Operating lease liability',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='operating_lease_right_of_use_asset',
            synonyms=[
                'OperatingLeaseRightOfUseAsset',
                'RightOfUseAssetObtainedInExchangeForOperatingLeaseLiability',
            ],
            description='Operating lease right-of-use asset',
            category='balance_sheet'
        ),
        SynonymGroup(
            name='finance_lease_liability',
            synonyms=[
                'FinanceLeaseLiability',
                'FinanceLeaseLiabilityCurrent',
                'FinanceLeaseLiabilityNoncurrent',
                'CapitalLeaseObligations',
            ],
            description='Finance/capital lease liability',
            category='balance_sheet'
        ),
        
        # ═══════════════════════════════════════════════════════════════════
        # FINANCIAL RATIOS / METRICS
        # ═══════════════════════════════════════════════════════════════════
        SynonymGroup(
            name='book_value_per_share',
            synonyms=[
                'BookValuePerShare',
                'BookValuePerShareCommon',
            ],
            description='Book value per share',
            category='metrics'
        ),
        SynonymGroup(
            name='return_on_equity',
            synonyms=[
                'ReturnOnEquity',
                'ROE',
            ],
            description='Return on equity',
            category='metrics'
        ),
        SynonymGroup(
            name='return_on_assets',
            synonyms=[
                'ReturnOnAssets',
                'ROA',
            ],
            description='Return on assets',
            category='metrics'
        ),
    ]
    return _builtin_groups_cache


class SynonymGroups:
    """
    Centralized manager for XBRL tag synonym groups.
    
    Provides a unified interface for managing synonym groups that can be used
    across all financial statement classes. This is the foundation for the
    shared standardization infrastructure.
    
    The manager comes pre-loaded with 40+ common financial concept groups
    (revenue, net_income, capex, etc.) and supports user-defined custom groups.
    
    Example:
    >>> synonyms = SynonymGroups()
    >>>
    >>> # Get pre-built group
    >>> revenue = synonyms.get_group('revenue')
    >>> print(revenue.synonyms[:3])
    ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'Revenue']
    >>>
    >>> # Identify concept from tag
    >>> info = synonyms.identify_concept('NetIncomeLoss')
    >>> print(info.name)
    'net_income'
    >>>
    >>> # Register custom group
    >>> synonyms.register_group(
    ...     name='my_revenue',
    ...     synonyms=['CustomRevenue1', 'CustomRevenue2']
    ... )
    
    Attributes:
        _groups: Dictionary of name -> SynonymGroup
        _tag_index: Reverse index of tag -> group name for fast lookups
    """
    
    def __init__(self, load_builtin: bool = True):
        """
        Initialize SynonymGroups manager.
        
        Args:
            load_builtin: Whether to load pre-built synonym groups (default: True)
        """
        self._groups: Dict[str, SynonymGroup] = {}
        self._tag_index: Dict[str, List[str]] = {}  # tag -> [group_name1, group_name2, ...]
        self._user_groups: Dict[str, SynonymGroup] = {}  # Track user-defined groups
        
        if load_builtin:
            self._load_builtin_groups()
    
    def _load_builtin_groups(self) -> None:
        """Load pre-built synonym groups for common financial concepts."""
        builtin_groups = _get_builtin_groups_cached()
        for group in builtin_groups:
            self._register_group_internal(group, is_user_defined=False)
    
    def _register_group_internal(self, group: SynonymGroup, is_user_defined: bool = False) -> None:
        """
        Internal method to register a group and update indices.
        
        Tags can belong to multiple groups (multi-group membership). This allows
        concepts like DepreciationAndAmortization to appear in both income_statement
        and cash_flow contexts.
        
        Args:
            group: The SynonymGroup to register
            is_user_defined: Whether this is a user-defined group
        """
        self._groups[group.name] = group
        
        if is_user_defined:
            self._user_groups[group.name] = group
        
        # Update reverse index - append to list to support multi-group membership
        for tag in group.synonyms:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = []
            # Avoid duplicates if same group is re-registered
            if group.name not in self._tag_index[tag_lower]:
                self._tag_index[tag_lower].append(group.name)
    
    def get_group(self, name: str) -> Optional[SynonymGroup]:
        """
        Get a synonym group by name.
        
        Args:
            name: The canonical name of the concept (e.g., 'revenue', 'net_income')
            
        Returns:
            SynonymGroup if found, None otherwise
            
        Example:
        >>> synonyms = SynonymGroups()
        >>> revenue = synonyms.get_group('revenue')
        >>> print(revenue.synonyms[:2])
        ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues']
        """
        normalized = _normalize_name(name)
        return self._groups.get(normalized)
    
    def get_synonyms(self, name: str) -> List[str]:
        """
        Get the list of synonyms for a concept.
        
        Convenience method that returns just the synonym list.
        
        Args:
            name: The canonical name of the concept
            
        Returns:
            List of synonym tags, or empty list if not found
            
        Example:
        >>> synonyms = SynonymGroups()
        >>> tags = synonyms.get_synonyms('capex')
        >>> print(tags[:2])
        ['PaymentsToAcquirePropertyPlantAndEquipment', 'CapitalExpenditures']
        """
        group = self.get_group(name)
        return group.synonyms if group else []
    
    def identify_concept(self, tag: str, context: Optional[Dict[str, Any]] = None) -> Optional[ConceptInfo]:
        """
        Identify which concept a tag belongs to (returns first match).
        
        Performs reverse lookup to find the canonical concept name
        for a given XBRL tag. If the tag belongs to multiple groups,
        returns the first one (order of registration).
        
        For tags that may belong to multiple concepts, use identify_concepts()
        to get all matches.
        
        Args:
            tag: XBRL tag to identify (with or without namespace prefix)
            context: Optional context for disambiguation (section, statement_type, etc.)
            
        Returns:
            ConceptInfo if tag is recognized, None otherwise
            
        Example:
        >>> synonyms = SynonymGroups()
        >>> info = synonyms.identify_concept('us-gaap:NetIncomeLoss')
        >>> print(info.name)
        'net_income'
        >>> print(info.description)
        'Net income/loss'
        """
        # Try reverse index first (if available)
        try:
            from financial4all.xbrl.standardization.reverse_index import get_reverse_index
            reverse_index = get_reverse_index()
            standard_concept = reverse_index.get_standard_concept(tag, context)
            if standard_concept:
                # Find matching group
                normalized_concept = standard_concept.lower().replace(' ', '_').replace('-', '_')
                group = self._groups.get(normalized_concept)
                if group:
                    return ConceptInfo(
                        name=normalized_concept,
                        tag=tag,
                        group=group,
                        match_type='exact'
                    )
        except (ImportError, AttributeError):
            pass
        
        # Fallback to existing logic
        # Normalize tag
        normalized = SynonymGroup._strip_namespace(tag).lower()
        
        # Look up in index - returns list of group names
        group_names = self._tag_index.get(normalized, [])
        if group_names:
            group_name = group_names[0]  # Return first match
            group = self._groups[group_name]
            return ConceptInfo(
                name=group_name,
                tag=tag,
                group=group,
                match_type='exact'
            )
        
        return None
    
    def identify_concepts(self, tag: str) -> List[ConceptInfo]:
        """
        Identify all concepts a tag belongs to.
        
        Performs reverse lookup to find all canonical concept names
        for a given XBRL tag. Tags can belong to multiple groups
        (multi-group membership) when they have different meanings
        in different contexts.
        
        Args:
            tag: XBRL tag to identify (with or without namespace prefix)
            
        Returns:
            List of ConceptInfo for all matching groups (empty if not recognized)
        """
        # Normalize tag
        normalized = SynonymGroup._strip_namespace(tag).lower()
        
        # Look up in index - returns list of group names
        group_names = self._tag_index.get(normalized, [])
        
        results = []
        for group_name in group_names:
            group = self._groups[group_name]
            results.append(ConceptInfo(
                name=group_name,
                tag=tag,
                group=group,
                match_type='exact'
            ))
        
        return results
    
    def register_group(
        self,
        name: str,
        synonyms: List[str],
        description: str = "",
        namespace: str = "us-gaap",
        priority_order: str = "listed",
        category: str = ""
    ) -> SynonymGroup:
        """
        Register a custom synonym group.
        
        User-defined groups take precedence over built-in groups
        if there are naming conflicts.
        
        Args:
            name: Canonical name for the concept
            synonyms: List of XBRL tags that represent this concept
            description: Human-readable description
            namespace: Default namespace for tags
            priority_order: How to order synonyms ('listed', 'frequency', 'specificity')
            category: Financial statement category
            
        Returns:
            The registered SynonymGroup
        """
        group = SynonymGroup(
            name=name,
            synonyms=synonyms,
            description=description,
            namespace=namespace,
            priority_order=priority_order,
            category=category
        )
        self._register_group_internal(group, is_user_defined=True)
        log.info(f"Registered custom synonym group: {group.name}")
        return group
    
    def unregister_group(self, name: str) -> bool:
        """
        Remove a user-defined synonym group.
        
        Only user-defined groups can be removed. Built-in groups
        cannot be unregistered.
        
        Args:
            name: Name of the group to remove
            
        Returns:
            True if group was removed, False if not found or is built-in
        """
        normalized = _normalize_name(name)
        
        if normalized not in self._user_groups:
            log.warning(f"Cannot unregister group '{name}': not a user-defined group")
            return False
        
        group = self._groups.pop(normalized, None)
        self._user_groups.pop(normalized, None)
        
        if group:
            # Remove from index - handle list-based index
            for tag in group.synonyms:
                tag_lower = tag.lower()
                if tag_lower in self._tag_index:
                    group_list = self._tag_index[tag_lower]
                    if normalized in group_list:
                        group_list.remove(normalized)
                    # Clean up empty lists
                    if not group_list:
                        del self._tag_index[tag_lower]
            return True
        
        return False
    
    def list_groups(self, category: Optional[str] = None) -> List[str]:
        """
        List all available synonym group names.
        
        Args:
            category: Optional filter by category (e.g., 'income_statement', 'balance_sheet')
            
        Returns:
            List of group names, sorted alphabetically
        """
        if category:
            return sorted([
                name for name, group in self._groups.items()
                if group.category == category
            ])
        return sorted(self._groups.keys())
    
    def export_to_json(self, file_path: Union[str, Path]) -> None:
        """
        Export user-defined groups to JSON file.
        
        Args:
            file_path: Path to JSON file
        """
        data = {
            'groups': [group.to_dict() for group in self._user_groups.values()]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        log.info(f"Exported {len(self._user_groups)} groups to {file_path}")
    
    def import_from_json(self, file_path: Union[str, Path]) -> None:
        """
        Import user-defined groups from JSON file.
        
        Args:
            file_path: Path to JSON file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        groups_imported = 0
        for group_dict in data.get('groups', []):
            group = SynonymGroup.from_dict(group_dict)
            self._register_group_internal(group, is_user_defined=True)
            groups_imported += 1
        
        log.info(f"Imported {groups_imported} groups from {file_path}")
    
    def get_standard_concept(self, tag: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get standard concept name for a tag using reverse index (if available).
        
        Args:
            tag: XBRL tag to look up
            context: Optional context for disambiguation
            
        Returns:
            Standard concept name or None
        """
        try:
            from financial4all.xbrl.standardization.reverse_index import get_reverse_index
            reverse_index = get_reverse_index()
            return reverse_index.get_standard_concept(tag, context)
        except (ImportError, AttributeError):
            # Fallback to identify_concept
            info = self.identify_concept(tag, context)
            return info.name if info else None
    
    def get_display_name(self, tag: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get user-friendly display name for a tag.
        
        Args:
            tag: XBRL tag to look up
            context: Optional context for disambiguation
            
        Returns:
            Display name or None
        """
        try:
            from financial4all.xbrl.standardization.reverse_index import get_reverse_index
            reverse_index = get_reverse_index()
            return reverse_index.get_display_name(tag, context)
        except (ImportError, AttributeError):
            # Fallback to concept name
            info = self.identify_concept(tag, context)
            if info:
                # Try to get display name from StandardConcept enum
                try:
                    from financial4all.xbrl.standardization.standard_concepts import StandardConcept
                    for concept in StandardConcept:
                        if concept.value.lower().replace(' ', '_').replace('-', '_') == info.name:
                            return concept.value
                except ImportError:
                    pass
                return info.name.replace('_', ' ').title()
            return None
    
    def is_ambiguous(self, tag: str) -> bool:
        """
        Check if a tag is ambiguous (maps to multiple concepts).
        
        Args:
            tag: XBRL tag to check
            
        Returns:
            True if ambiguous, False otherwise
        """
        try:
            from financial4all.xbrl.standardization.reverse_index import get_reverse_index
            reverse_index = get_reverse_index()
            return reverse_index.is_ambiguous(tag)
        except (ImportError, AttributeError):
            # Fallback: check if tag appears in multiple groups
            normalized = SynonymGroup._strip_namespace(tag).lower()
            group_names = self._tag_index.get(normalized, [])
            return len(group_names) > 1


def get_synonym_groups() -> SynonymGroups:
    """
    Get the default singleton SynonymGroups instance.
    
    Returns:
        Global SynonymGroups instance
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = SynonymGroups()
    return _default_instance


# ═══════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY LAYER
# ═══════════════════════════════════════════════════════════════════

class StandardizationStore:
    """
    Legacy compatibility layer for StandardizationStore.
    
    This class maintains backward compatibility with existing code
    while delegating to the new SynonymGroups system.
    
    Maps standardized concept names to XBRL concept names and vice versa.
    """
    
    def __init__(self):
        """Initialize standardization store."""
        self._synonym_groups = get_synonym_groups()
    
    def add_mapping(self, standard_name: str, xbrl_concepts: List[str]) -> None:
        """
        Add a standardization mapping.
        
        Args:
            standard_name: Standardized concept name
            xbrl_concepts: List of XBRL concept names that map to this standard name
        """
        # Convert to normalized name
        normalized = _normalize_name(standard_name)
        
        # Check if group already exists
        existing_group = self._synonym_groups.get_group(normalized)
        if existing_group:
            # Merge synonyms
            merged_synonyms = list(set(existing_group.synonyms + xbrl_concepts))
            # Unregister old and register new
            self._synonym_groups.unregister_group(normalized)
            self._synonym_groups.register_group(
                name=normalized,
                synonyms=merged_synonyms,
                description=existing_group.description,
                category=existing_group.category
            )
        else:
            # Create new group
            self._synonym_groups.register_group(
                name=normalized,
                synonyms=xbrl_concepts,
                description=f"Standardized mapping for {standard_name}",
                category=""
            )
    
    def get_standard_name(self, xbrl_concept: str) -> Optional[str]:
        """
        Get standardized name for an XBRL concept.
        
        Args:
            xbrl_concept: XBRL concept name
            
        Returns:
            Standardized name or None if not found
        """
        info = self._synonym_groups.identify_concept(xbrl_concept)
        return info.name if info else None
    
    def get_xbrl_concepts(self, standard_name: str) -> List[str]:
        """
        Get XBRL concept names for a standardized name.
        
        Args:
            standard_name: Standardized concept name
            
        Returns:
            List of XBRL concept names
        """
        return self._synonym_groups.get_synonyms(standard_name)


# Global standardization store instance (legacy)
_standardization_store: Optional[StandardizationStore] = None


def get_default_store() -> StandardizationStore:
    """
    Get the default standardization store instance (legacy compatibility).
    
    Returns:
        StandardizationStore instance
    """
    global _standardization_store
    if _standardization_store is None:
        _standardization_store = StandardizationStore()
    return _standardization_store
