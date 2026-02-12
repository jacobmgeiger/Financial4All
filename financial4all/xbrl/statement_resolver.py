# financial4all/xbrl/statement_resolver.py
"""
Statement Resolution for XBRL data (EdgarTools parity).

This module provides a robust system for identifying and matching XBRL financial
statements, notes, and disclosures regardless of taxonomy variations and
company-specific customizations.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from financial4all.core import log
from financial4all.xbrl.statements import statement_to_concepts


class StatementCategory(Enum):
    """Categories of XBRL presentation sections."""
    FINANCIAL_STATEMENT = "statement"
    NOTE = "note"
    DISCLOSURE = "disclosure"
    DOCUMENT = "document"  # For cover page, signatures, etc.
    OTHER = "other"


@dataclass
class ConceptPattern:
    """Pattern for matching statement concepts across different taxonomies."""
    pattern: str
    weight: float = 1.0


@dataclass
class StatementTypeInfo:
    """Detailed information about a statement type for matching (EdgarTools parity)."""
    name: str
    primary_concepts: List[str]
    category: StatementCategory = StatementCategory.FINANCIAL_STATEMENT
    alternative_concepts: List[str] = field(default_factory=list)
    concept_patterns: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    role_patterns: List[str] = field(default_factory=list)
    title: str = ""
    supports_parenthetical: bool = False
    weight_map: Dict[str, float] = field(default_factory=dict)

    def match_concept(self, concept_name: str) -> bool:
        """Check if a concept name matches this statement type's concepts."""
        if concept_name in self.primary_concepts:
            return True
        if concept_name in self.alternative_concepts:
            return True
        for pattern in self.concept_patterns:
            if re.match(pattern, concept_name):
                return True
        return False

    def match_role(self, role_uri: str, role_name: str = "", role_def: str = "") -> bool:
        """Check if role information matches this statement type."""
        name_lower = self.name.lower()
        if name_lower in role_uri.lower():
            return True
        if role_name and name_lower in role_name.lower():
            return True
        if role_def and name_lower in role_def.lower():
            return True
        for pattern in self.role_patterns:
            if re.match(pattern, role_uri) or (role_name and re.match(pattern, role_name)):
                return True
        return False


# Registry of statement types with matching information (EdgarTools parity)
statement_registry = {
    "BalanceSheet": StatementTypeInfo(
        name="BalanceSheet",
        category=StatementCategory.FINANCIAL_STATEMENT,
        primary_concepts=["us-gaap_StatementOfFinancialPositionAbstract"],
        alternative_concepts=[
            "us-gaap_BalanceSheetAbstract",
            "ifrs-full_StatementOfFinancialPositionAbstract",
        ],
        concept_patterns=[
            r".*_StatementOfFinancialPositionAbstract$",
            r".*_BalanceSheetAbstract$",
            r".*_ConsolidatedBalanceSheetsAbstract$",
            r".*_CondensedConsolidatedBalanceSheetsUnauditedAbstract$",
        ],
        key_concepts=[
            "us-gaap_Assets", "us-gaap_Liabilities", "us-gaap_StockholdersEquity",
            "ifrs-full_Assets", "ifrs-full_Liabilities", "ifrs-full_Equity",
        ],
        role_patterns=[
            r".*[Bb]alance[Ss]heet.*",
            r".*[Ss]tatement[Oo]f[Ff]inancial[Pp]osition.*",
            r".*StatementConsolidatedBalanceSheets.*",
        ],
        title="Consolidated Balance Sheets",
        supports_parenthetical=True,
        weight_map={"assets": 0.3, "liabilities": 0.3, "equity": 0.4},
    ),
    "IncomeStatement": StatementTypeInfo(
        name="IncomeStatement",
        category=StatementCategory.FINANCIAL_STATEMENT,
        primary_concepts=["us-gaap_IncomeStatementAbstract"],
        alternative_concepts=[
            "us-gaap_StatementOfIncomeAbstract",
            "ifrs-full_IncomeStatementAbstract",
            "ifrs-full_StatementOfComprehensiveIncomeAbstract",
            "ifrs-full_StatementOfProfitOrLossAbstract",
        ],
        concept_patterns=[
            r".*_IncomeStatementAbstract$",
            r".*_StatementOfIncomeAbstract$",
            r".*_ConsolidatedStatementsOfIncomeAbstract$",
            r".*_CondensedConsolidatedStatementsOfIncomeUnauditedAbstract$",
        ],
        key_concepts=[
            "us-gaap_Revenues", "us-gaap_NetIncomeLoss",
            "ifrs-full_Revenue", "ifrs-full_ProfitLoss",
        ],
        role_patterns=[
            r".*[Ii]ncome[Ss]tatements?.*",
            r".*[Ss]tatements?[Oo]f[Ii]ncome.*",
            r".*[Ss]tatements?[Oo]f[Oo]perations.*",
            r".*StatementConsolidatedStatementsOfIncome.*",
        ],
        title="Consolidated Statement of Income",
        supports_parenthetical=True,
        weight_map={"revenues": 0.4, "netIncomeLoss": 0.6},
    ),
    "CashFlowStatement": StatementTypeInfo(
        name="CashFlowStatement",
        category=StatementCategory.FINANCIAL_STATEMENT,
        primary_concepts=["us-gaap_StatementOfCashFlowsAbstract"],
        alternative_concepts=["ifrs-full_StatementOfCashFlowsAbstract"],
        concept_patterns=[
            r".*_StatementOfCashFlowsAbstract$",
            r".*_CashFlowsAbstract$",
            r".*_ConsolidatedStatementsOfCashFlowsAbstract$",
            r".*_CondensedConsolidatedStatementsOfCashFlowsUnauditedAbstract$",
        ],
        key_concepts=[
            "us-gaap_NetCashProvidedByUsedInOperatingActivities",
            "us-gaap_CashAndCashEquivalentsPeriodIncreaseDecrease",
            "ifrs-full_CashFlowsFromUsedInOperatingActivities",
            "ifrs-full_IncreaseDecreaseInCashAndCashEquivalents",
        ],
        role_patterns=[
            r".*[Cc]ash[Ff]low.*",
            r".*[Ss]tatement[Oo]f[Cc]ash[Ff]lows.*",
            r".*StatementConsolidatedStatementsOfCashFlows.*",
        ],
        title="Consolidated Statement of Cash Flows",
        supports_parenthetical=False,
    ),
    "StatementOfEquity": StatementTypeInfo(
        name="StatementOfEquity",
        category=StatementCategory.FINANCIAL_STATEMENT,
        primary_concepts=["us-gaap_StatementOfStockholdersEquityAbstract"],
        alternative_concepts=[
            "us-gaap_StatementOfShareholdersEquityAbstract",
            "us-gaap_StatementOfPartnersCapitalAbstract",
            "us-gaap_IncreaseDecreaseInStockholdersEquityRollForward",
            "ifrs-full_StatementOfChangesInEquityAbstract",
        ],
        concept_patterns=[
            r".*_StatementOfStockholdersEquityAbstract$",
            r".*_StatementOfShareholdersEquityAbstract$",
            r".*_StatementOfChangesInEquityAbstract$",
            r".*_ConsolidatedStatementsOfShareholdersEquityAbstract$",
            r".*_IncreaseDecreaseInStockholdersEquityRollForward$",
        ],
        key_concepts=[
            "us-gaap_StockholdersEquity", "us-gaap_CommonStock", "us-gaap_RetainedEarnings",
            "ifrs-full_Equity", "ifrs-full_IssuedCapital", "ifrs-full_RetainedEarnings",
        ],
        role_patterns=[
            r".*[Ee]quity.*",
            r".*[Ss]tockholders.*",
            r".*[Ss]hareholders.*",
            r".*[Cc]hanges[Ii]n[Ee]quity.*",
            r".*StatementConsolidatedStatementsOfStockholdersEquity.*",
        ],
        title="Consolidated Statement of Equity",
        supports_parenthetical=True,
    ),
    "ComprehensiveIncome": StatementTypeInfo(
        name="ComprehensiveIncome",
        category=StatementCategory.FINANCIAL_STATEMENT,
        primary_concepts=["us-gaap_StatementOfIncomeAndComprehensiveIncomeAbstract"],
        alternative_concepts=[
            "us-gaap_StatementOfComprehensiveIncomeAbstract",
            "ifrs-full_StatementOfComprehensiveIncomeAbstract",
            "ifrs-full_StatementOfProfitOrLossAndOtherComprehensiveIncomeAbstract",
        ],
        concept_patterns=[
            r".*_ComprehensiveIncomeAbstract$",
            r".*_StatementOfComprehensiveIncomeAbstract$",
            r".*_ConsolidatedStatementsOfComprehensiveIncomeAbstract$",
        ],
        key_concepts=[
            "us-gaap_ComprehensiveIncomeNetOfTax",
            "ifrs-full_ComprehensiveIncome",
            "ifrs-full_OtherComprehensiveIncome",
        ],
        role_patterns=[
            r".*[Cc]omprehensive[Ii]ncome.*",
            r".*[Oo]ther[Cc]omprehensive.*",
            r".*StatementConsolidatedStatementsOfComprehensiveIncome.*",
        ],
        title="Consolidated Statement of Comprehensive Income",
        supports_parenthetical=True,
    ),
    "Notes": StatementTypeInfo(
        name="Notes",
        category=StatementCategory.NOTE,
        primary_concepts=["us-gaap_NotesToFinancialStatementsAbstract"],
        alternative_concepts=[],
        concept_patterns=[
            r".*_NotesToFinancialStatementsAbstract$",
            r".*_NotesAbstract$",
        ],
        key_concepts=[],
        role_patterns=[
            r".*[Nn]otes[Tt]o[Ff]inancial[Ss]tatements.*",
            r".*[Nn]ote\s+\d+.*",
            r".*[Nn]otes.*",
        ],
        title="Notes to Financial Statements",
        supports_parenthetical=False,
    ),
    "AccountingPolicies": StatementTypeInfo(
        name="AccountingPolicies",
        category=StatementCategory.NOTE,
        primary_concepts=["us-gaap_AccountingPoliciesAbstract"],
        alternative_concepts=[],
        concept_patterns=[
            r".*_AccountingPoliciesAbstract$",
            r".*_SignificantAccountingPoliciesAbstract$",
        ],
        key_concepts=["us-gaap_SignificantAccountingPoliciesTextBlock"],
        role_patterns=[
            r".*[Aa]ccounting[Pp]olicies.*",
            r".*[Ss]ignificant[Aa]ccounting[Pp]olicies.*",
        ],
        title="Significant Accounting Policies",
        supports_parenthetical=False,
    ),
    "Disclosures": StatementTypeInfo(
        name="Disclosures",
        category=StatementCategory.DISCLOSURE,
        primary_concepts=["us-gaap_DisclosuresAbstract"],
        alternative_concepts=[],
        concept_patterns=[
            r".*_DisclosuresAbstract$",
            r".*_DisclosureAbstract$",
        ],
        key_concepts=[],
        role_patterns=[r".*[Dd]isclosure.*"],
        title="Disclosures",
        supports_parenthetical=False,
    ),
    "SegmentDisclosure": StatementTypeInfo(
        name="SegmentDisclosure",
        category=StatementCategory.DISCLOSURE,
        primary_concepts=["us-gaap_SegmentDisclosureAbstract"],
        alternative_concepts=[],
        concept_patterns=[
            r".*_SegmentDisclosureAbstract$",
            r".*_SegmentReportingDisclosureAbstract$",
        ],
        key_concepts=["us-gaap_SegmentReportingDisclosureTextBlock"],
        role_patterns=[
            r".*[Ss]egment.*",
            r".*[Ss]egment[Rr]eporting.*",
            r".*[Ss]egment[Ii]nformation.*",
        ],
        title="Segment Information",
        supports_parenthetical=False,
    ),
    "CoverPage": StatementTypeInfo(
        name="CoverPage",
        category=StatementCategory.DOCUMENT,
        primary_concepts=["dei_CoverAbstract"],
        concept_patterns=[r".*_CoverAbstract$"],
        key_concepts=["dei_EntityRegistrantName", "dei_DocumentType"],
        role_patterns=[r".*[Cc]over.*"],
        title="Cover Page",
        supports_parenthetical=False,
    ),
    "ScheduleOfInvestments": StatementTypeInfo(
        name="ScheduleOfInvestments",
        category=StatementCategory.FINANCIAL_STATEMENT,
        primary_concepts=["us-gaap_ScheduleOfInvestmentsAbstract"],
        alternative_concepts=[
            "us-gaap_InvestmentsDebtAndEquitySecuritiesAbstract",
            "us-gaap_InvestmentHoldingsAbstract",
        ],
        concept_patterns=[
            r".*_ScheduleOfInvestmentsAbstract$",
            r".*_ConsolidatedScheduleofInvestmentsAbstract$",
            r".*_InvestmentHoldingsAbstract$",
        ],
        key_concepts=[
            "us-gaap_InvestmentOwnedAtFairValue",
            "us-gaap_InvestmentOwnedAtCost",
            "us-gaap_InvestmentOwnedBalancePrincipalAmount",
            "us-gaap_InvestmentOwnedBalanceShares",
            "us-gaap_InvestmentOwnedPercentOfNetAssets",
            "us-gaap_ScheduleOfInvestmentsLineItems",
        ],
        role_patterns=[
            r".*[Ss]chedule[Oo]f[Ii]nvestments.*",
            r".*[Cc]onsolidated[Ss]chedule[Oo]f[Ii]nvestments.*",
            r".*[Ii]nvestment[Hh]oldings.*",
            r".*[Pp]ortfolio[Ii]nvestments.*",
        ],
        title="Consolidated Schedule of Investments",
        supports_parenthetical=True,
    ),
    "FinancialHighlights": StatementTypeInfo(
        name="FinancialHighlights",
        category=StatementCategory.FINANCIAL_STATEMENT,
        primary_concepts=["us-gaap_InvestmentCompanyFinancialHighlightsAbstract"],
        alternative_concepts=["us-gaap_InvestmentCompanyAbstract"],
        concept_patterns=[
            r".*_FinancialHighlightsAbstract$",
            r".*_InvestmentCompanyFinancialHighlightsAbstract$",
        ],
        key_concepts=[
            "us-gaap_NetAssetValuePerShare",
            "us-gaap_InvestmentCompanyNetAssets",
            "us-gaap_InvestmentCompanyTotalReturn",
            "us-gaap_InvestmentCompanyExpenseRatio",
        ],
        role_patterns=[
            r".*[Ff]inancial[Hh]ighlights.*",
            r".*[Ii]nvestment[Cc]ompany[Ff]inancial[Hh]ighlights.*",
        ],
        title="Financial Highlights",
        supports_parenthetical=False,
    ),
}

# Mapping from StatementType enum snake_case values to PascalCase registry keys
_ENUM_TO_REGISTRY: Dict[str, str] = {
    "income_statement": "IncomeStatement",
    "balance_sheet": "BalanceSheet",
    "cash_flow_statement": "CashFlowStatement",
    "changes_in_equity": "StatementOfEquity",
    "statement_of_equity": "StatementOfEquity",
    "comprehensive_income": "ComprehensiveIncome",
    "segment_reporting": "SegmentDisclosure",
    "footnotes": "Notes",
    "accounting_policies": "AccountingPolicies",
}

# Essential concepts for validation (EdgarTools parity)
ESSENTIAL_CONCEPTS = {
    "IncomeStatement": {
        "revenue": [
            "us-gaap_Revenues",
            "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap_SalesRevenueNet",
            "us-gaap_NetSales",
            "us-gaap_TotalRevenuesAndOtherIncome",
            "ifrs-full_Revenue",
        ],
        "net_income": [
            "us-gaap_NetIncomeLoss",
            "us-gaap_ProfitLoss",
            "us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic",
            "ifrs-full_ProfitLoss",
            "ifrs-full_ProfitLossAttributableToOwnersOfParent",
        ],
    },
    "BalanceSheet": {
        "assets": [
            "us-gaap_Assets",
            "us-gaap_AssetsCurrent",
            "us-gaap_AssetsNoncurrent",
            "ifrs-full_Assets",
            "ifrs-full_CurrentAssets",
            "ifrs-full_NoncurrentAssets",
        ],
        "liabilities_or_equity": [
            "us-gaap_Liabilities",
            "us-gaap_StockholdersEquity",
            "us-gaap_LiabilitiesAndStockholdersEquity",
            "ifrs-full_Liabilities",
            "ifrs-full_Equity",
            "ifrs-full_EquityAndLiabilities",
        ],
    },
    "CashFlowStatement": {
        "operating": [
            "us-gaap_NetCashProvidedByUsedInOperatingActivities",
            "us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            "ifrs-full_CashFlowsFromUsedInOperatingActivities",
        ],
        "cash_change": [
            "us-gaap_CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
            "us-gaap_CashAndCashEquivalentsPeriodIncreaseDecrease",
            "ifrs-full_IncreaseDecreaseInCashAndCashEquivalents",
        ],
    },
    "ComprehensiveIncome": {
        "comprehensive_income": [
            "us-gaap_ComprehensiveIncomeNetOfTax",
            "us-gaap_ComprehensiveIncomeNetOfTaxIncludingPortionAttributableToNoncontrollingInterest",
            "ifrs-full_ComprehensiveIncome",
        ],
    },
    "StatementOfEquity": {
        "equity": [
            "us-gaap_StockholdersEquity",
            "us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "ifrs-full_Equity",
        ],
    },
}

VALIDATION_THRESHOLD = 0.5


class StatementType(Enum):
    """Statement type enum for backward compatibility (PascalCase values)."""
    INCOME_STATEMENT = "IncomeStatement"
    BALANCE_SHEET = "BalanceSheet"
    CASH_FLOW_STATEMENT = "CashFlowStatement"
    STATEMENT_OF_EQUITY = "StatementOfEquity"
    COMPREHENSIVE_INCOME = "ComprehensiveIncome"


class StatementResolver:
    """
    Resolves statement identifiers to actual XBRL statement roles.

    EdgarTools parity: multi-layered approach to statement matching,
    handling taxonomy variations and company-specific customizations.
    """

    def __init__(self, xbrl: Any):
        """
        Initialize with an XBRL object.

        Args:
            xbrl: XBRL object containing parsed data (presentation_trees, etc.)
        """
        self.xbrl = xbrl
        self._cache: Dict[str, Any] = {}
        self._statement_by_role_uri: Dict[str, Dict[str, Any]] = {}
        self._statement_by_role_name: Dict[str, List[Dict[str, Any]]] = {}
        self._statement_by_primary_concept: Dict[str, List[Dict[str, Any]]] = {}
        self._statement_by_type: Dict[str, List[Dict[str, Any]]] = {}
        self._statement_by_role_def: Dict[str, List[Dict[str, Any]]] = {}

        # Map legacy statement types to registry
        self._legacy_to_registry: Dict[str, str] = {}
        for legacy_type, info in statement_to_concepts.items():
            if legacy_type in statement_registry:
                self._legacy_to_registry[legacy_type] = legacy_type
                continue
            for reg_type, reg_info in statement_registry.items():
                if info.concept in reg_info.primary_concepts or info.concept in reg_info.alternative_concepts:
                    self._legacy_to_registry[legacy_type] = reg_type
                    break

        self._initialize_indices()

    def _initialize_indices(self) -> None:
        """Build lookup indices for fast statement retrieval."""
        statements = self.xbrl.get_all_statements()
        self._statement_by_role_uri = {}
        self._statement_by_role_name = {}
        self._statement_by_primary_concept = {}
        self._statement_by_type = {}
        self._statement_by_role_def = {}

        for stmt in statements:
            role = stmt.get("role", "")
            role_name = (stmt.get("role_name", "") or "").lower()
            primary_concept = stmt.get("primary_concept", "")
            stmt_type = stmt.get("type", "")
            role_def = (stmt.get("definition", "") or "").lower()

            self._statement_by_role_uri[role] = stmt

            if role_name:
                if role_name not in self._statement_by_role_name:
                    self._statement_by_role_name[role_name] = []
                self._statement_by_role_name[role_name].append(stmt)

            if primary_concept:
                if primary_concept not in self._statement_by_primary_concept:
                    self._statement_by_primary_concept[primary_concept] = []
                self._statement_by_primary_concept[primary_concept].append(stmt)

            if stmt_type:
                if stmt_type not in self._statement_by_type:
                    self._statement_by_type[stmt_type] = []
                self._statement_by_type[stmt_type].append(stmt)

            if role_def:
                def_key = role_def.replace(" ", "")
                if def_key not in self._statement_by_role_def:
                    self._statement_by_role_def[def_key] = []
                self._statement_by_role_def[def_key].append(stmt)

    def _validate_statement(
        self, stmt: Dict[str, Any], statement_type: str
    ) -> Tuple[bool, float, str]:
        """
        Validate that a resolved statement contains expected essential concepts.
        """
        if statement_type not in ESSENTIAL_CONCEPTS:
            return True, 1.0, "No validation rules defined"

        essential_groups = ESSENTIAL_CONCEPTS[statement_type]
        role = stmt.get("role", "")

        if role not in self.xbrl.presentation_trees:
            return False, 0.0, f"Role {role} not found in presentation trees"

        tree = self.xbrl.presentation_trees[role]
        all_nodes = set(tree.all_nodes.keys())

        groups_satisfied = 0
        total_groups = len(essential_groups)
        missing_groups = []

        for group_name, concepts in essential_groups.items():
            group_found = False
            for concept in concepts:
                normalized = concept.replace(":", "_")
                if concept in all_nodes or normalized in all_nodes:
                    group_found = True
                    break
            if group_found:
                groups_satisfied += 1
            else:
                missing_groups.append(group_name)

        confidence = groups_satisfied / total_groups if total_groups > 0 else 1.0
        is_valid = confidence >= VALIDATION_THRESHOLD

        if is_valid:
            reason = (
                "All essential concept groups present"
                if confidence == 1.0
                else f"Validation passed ({groups_satisfied}/{total_groups} groups): missing {missing_groups}"
            )
        else:
            reason = (
                f"Validation failed ({groups_satisfied}/{total_groups} groups): "
                f"missing {missing_groups}"
            )
        return is_valid, confidence, reason

    def _match_by_primary_concept(
        self, statement_type: str, is_parenthetical: bool = False
    ) -> Tuple[List[Dict[str, Any]], Optional[str], float]:
        """Match statements using primary concept names."""
        registry_type = self._legacy_to_registry.get(statement_type, statement_type)
        if registry_type not in statement_registry:
            return [], None, 0.0

        registry_entry = statement_registry[registry_type]
        matched_statements = []

        for concept in registry_entry.primary_concepts + registry_entry.alternative_concepts:
            if concept in self._statement_by_primary_concept:
                for stmt in self._statement_by_primary_concept[concept]:
                    if registry_entry.supports_parenthetical:
                        role_def = (stmt.get("definition", "") or "").lower()
                        is_role_parenthetical = "parenthetical" in role_def
                        if is_parenthetical != is_role_parenthetical:
                            continue
                    matched_statements.append(stmt)

        if matched_statements:
            matched_statements.sort(
                key=lambda s: self._score_statement_quality(s, statement_type),
                reverse=True,
            )
            return matched_statements, matched_statements[0]["role"], 0.9
        return [], None, 0.0

    def _match_by_concept_pattern(
        self, statement_type: str, is_parenthetical: bool = False
    ) -> Tuple[List[Dict[str, Any]], Optional[str], float]:
        """Match statements using regex patterns on concept names."""
        registry_type = self._legacy_to_registry.get(statement_type, statement_type)
        if registry_type not in statement_registry:
            return [], None, 0.0

        registry_entry = statement_registry[registry_type]
        if not registry_entry.concept_patterns:
            return [], None, 0.0

        all_statements = self.xbrl.get_all_statements()
        matched_statements = []

        for stmt in all_statements:
            primary_concept = stmt.get("primary_concept", "")
            if not primary_concept:
                continue
            for pattern in registry_entry.concept_patterns:
                if re.match(pattern, primary_concept):
                    if registry_entry.supports_parenthetical:
                        role_def = (stmt.get("definition", "") or "").lower()
                        is_role_parenthetical = "parenthetical" in role_def
                        if is_parenthetical != is_role_parenthetical:
                            continue
                    matched_statements.append(stmt)
                    break

        if matched_statements:
            matched_statements.sort(
                key=lambda s: self._score_statement_quality(s, statement_type),
                reverse=True,
            )
            return matched_statements, matched_statements[0]["role"], 0.85
        return [], None, 0.0

    def _score_statement_quality(self, stmt: Dict[str, Any], statement_type: str = "") -> int:
        """Score a statement to prefer complete financial statements over fragments."""
        score = 100
        role_def = (stmt.get("definition", "") or "").lower()
        role_uri = (stmt.get("role", "") or "").lower()

        fragment_keywords = [
            "details", "detail", "tables", "table", "schedule", "schedules",
            "textual", "narrative", "policy", "policies", "disclosure",
            "supplemental", "additional", "breakdown", "summary",
        ]
        for keyword in fragment_keywords:
            if keyword in role_def or keyword in role_uri:
                score -= 50
                break

        if statement_type == "IncomeStatement":
            clean_def = role_def.replace(" ", "").replace("-", "").replace("_", "")
            clean_uri = role_uri.replace(" ", "").replace("-", "").replace("_", "")
            operations_indicators = [
                "operations", "statementsofincome", "statementsofearnings",
                "incomestatement", "operationsand",
            ]
            is_combined_statement = any(
                ind in clean_def or ind in clean_uri for ind in operations_indicators
            )
            if not is_combined_statement:
                comprehensive_indicators = ["comprehensiveincome", "othercomprehensive"]
                for indicator in comprehensive_indicators:
                    if indicator in clean_def or indicator in clean_uri:
                        score -= 100
                        break

            tax_indicators = [
                "incometax", "taxbenefit", "taxprovision",
                "taxexpense", "deferredtax",
            ]
            for indicator in tax_indicators:
                if indicator in clean_def or indicator in clean_uri:
                    score -= 100
                    break

        if "parenthetical" in role_def or "parenthetical" in role_uri:
            score -= 80

        if "consolidated" in role_def or "consolidated" in role_uri:
            score += 30
        if "condensed" in role_def or "condensed" in role_uri:
            score += 20

        primary_names = [
            "consolidatedbalancesheets",
            "consolidatedstatementsofoperations",
            "consolidatedstatementsofincome",
            "consolidatedstatementsofcashflows",
            "consolidatedstatementsofequity",
            "consolidatedstatementsofstockholdersequity",
        ]
        clean_def = role_def.replace(" ", "").replace("-", "").replace("_", "")
        if clean_def in primary_names:
            score += 50

        if statement_type in ESSENTIAL_CONCEPTS:
            is_valid, validation_conf, reason = self._validate_statement(stmt, statement_type)
            if is_valid:
                score += int(validation_conf * 30)
            else:
                score -= 50
                log.debug("Statement validation failed for %s: %s", statement_type, reason)

        return score

    def _match_by_role_pattern(
        self, statement_type: str, is_parenthetical: bool = False
    ) -> Tuple[List[Dict[str, Any]], Optional[str], float]:
        """Match statements using role URI or role name patterns."""
        registry_type = self._legacy_to_registry.get(statement_type, statement_type)
        if registry_type not in statement_registry:
            return [], None, 0.0

        registry_entry = statement_registry[registry_type]
        if not registry_entry.role_patterns:
            return [], None, 0.0

        all_statements = self.xbrl.get_all_statements()
        matched_statements = []

        for stmt in all_statements:
            role = stmt.get("role", "")
            role_name = stmt.get("role_name", "")
            for pattern in registry_entry.role_patterns:
                if re.search(pattern, role, re.IGNORECASE) or (
                    role_name and re.search(pattern, role_name, re.IGNORECASE)
                ):
                    if registry_entry.supports_parenthetical:
                        role_def = (stmt.get("definition", "") or "").lower()
                        is_role_parenthetical = "parenthetical" in role_def
                        if is_parenthetical != is_role_parenthetical:
                            continue
                    matched_statements.append(stmt)
                    break

        if matched_statements:
            matched_statements.sort(
                key=lambda s: self._score_statement_quality(s, statement_type),
                reverse=True,
            )
            return matched_statements, matched_statements[0]["role"], 0.7
        return [], None, 0.0

    def find_statement(
        self,
        role_or_type: str,
        is_parenthetical: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
        """
        Find statement by role URI, statement type, or short name.

        Args:
            role_or_type: Role URI, statement type (e.g. 'IncomeStatement'), or short name
            is_parenthetical: Whether to look for a parenthetical statement

        Returns:
            Tuple of (matching_statements, found_role, actual_statement_type)
        """
        # Normalize enum-style input
        role_or_type_normalized = role_or_type
        if hasattr(role_or_type, "value"):
            role_or_type_normalized = getattr(role_or_type, "value", str(role_or_type))
        if role_or_type_normalized in _ENUM_TO_REGISTRY:
            role_or_type_normalized = _ENUM_TO_REGISTRY[role_or_type_normalized]

        # 1. Direct role URI match
        if role_or_type in self._statement_by_role_uri:
            stmt = self._statement_by_role_uri[role_or_type]
            actual_type = stmt.get("type")
            return [stmt], role_or_type, actual_type

        # 2. Direct statement type match
        if role_or_type_normalized in self._statement_by_type:
            matched = self._statement_by_type[role_or_type_normalized]
            matched.sort(
                key=lambda s: self._score_statement_quality(s, role_or_type_normalized),
                reverse=True,
            )
            stmt = matched[0]
            return matched, stmt["role"], stmt.get("type")

        # 3. Match by primary concept
        matched, role, _ = self._match_by_primary_concept(
            role_or_type_normalized, is_parenthetical
        )
        if role:
            return matched, role, matched[0].get("type") if matched else None

        # 4. Match by concept pattern
        matched, role, _ = self._match_by_concept_pattern(
            role_or_type_normalized, is_parenthetical
        )
        if role:
            return matched, role, matched[0].get("type") if matched else None

        # 5. Match by role pattern
        matched, role, _ = self._match_by_role_pattern(
            role_or_type_normalized, is_parenthetical
        )
        if role:
            return matched, role, matched[0].get("type") if matched else None

        return [], None, None
