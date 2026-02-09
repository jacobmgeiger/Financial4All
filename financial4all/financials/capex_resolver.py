# financial4all/financials/capex_resolver.py
"""
CapEx resolution module.

This module provides a centralized, DRY implementation for resolving
Capital Expenditures (CapEx) from XBRL data using multiple strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from financial4all.xbrl.facts import FactSet
from financial4all.core import log


class CapExValidator:
    """
    Validates CapEx values against accounting relationships and historical patterns.
    Lives in capex_resolver to avoid circular import with cash_flow.
    """
    MAX_CAPEX_TO_PPE_CHANGE_RATIO = 5.0
    MAX_CAPEX_TO_DA_RATIO = 15.0
    MAX_CAPEX_PCT_OF_SALES = 0.50
    MAX_HISTORICAL_JUMP_RATIO = 4.0
    MAX_DA_TO_PPE_RATIO = 0.50
    MAX_DA_TO_REVENUE_RATIO = 0.20
    MAX_DA_HISTORICAL_JUMP_RATIO = 3.0
    CAPEX_PPE_TOLERANCE = 0.50

    @staticmethod
    def validate_capex_value(
        capex_value: float,
        period_key: str,
        ppe_change: Optional[float] = None,
        da_value: Optional[float] = None,
        revenue: Optional[float] = None,
        previous_capex: Optional[float] = None,
        ppe_value: Optional[float] = None
    ) -> Dict[str, Any]:
        issues = []
        confidence = 1.0
        abs_capex = abs(capex_value)
        suggested = None
        has_ppe_change = ppe_change is not None
        has_da = da_value is not None and da_value > 0
        has_revenue = revenue is not None and revenue > 0
        has_prev_capex = previous_capex is not None and previous_capex != 0
        if not has_ppe_change and not has_da and not has_revenue:
            confidence *= 0.5
            issues.append("No validation context available (PP&E change, D&A, or Revenue)")
        if ppe_change is not None:
            abs_ppe_change = abs(ppe_change)
            if abs_ppe_change > 0:
                ratio = abs_capex / abs_ppe_change
                if ratio > CapExValidator.MAX_CAPEX_TO_PPE_CHANGE_RATIO:
                    issues.append(
                        f"CapEx ({abs_capex:.0f}) is {ratio:.1f}x PP&E change ({abs_ppe_change:.0f}), "
                        f"exceeds typical threshold ({CapExValidator.MAX_CAPEX_TO_PPE_CHANGE_RATIO}x). "
                        f"May include business acquisitions or timing differences."
                    )
                    confidence *= 0.6
                elif ratio > CapExValidator.MAX_CAPEX_TO_PPE_CHANGE_RATIO * 0.7:
                    issues.append(
                        f"CapEx ({abs_capex:.0f}) is {ratio:.1f}x PP&E change ({abs_ppe_change:.0f}), "
                        f"above typical range ({CapExValidator.MAX_CAPEX_TO_PPE_CHANGE_RATIO}x)."
                    )
                    confidence *= 0.9
        if da_value is not None and da_value > 0:
            ratio = abs_capex / da_value
            if ratio > CapExValidator.MAX_CAPEX_TO_DA_RATIO:
                issues.append(
                    f"CapEx/D&A ratio ({ratio:.1f}x) exceeds typical threshold "
                    f"({CapExValidator.MAX_CAPEX_TO_DA_RATIO}x). May include acquisitions or reflect growth."
                )
                confidence *= 0.7
        if revenue is not None and revenue > 0:
            pct_of_sales = abs_capex / revenue
            if pct_of_sales > CapExValidator.MAX_CAPEX_PCT_OF_SALES:
                issues.append(
                    f"CapEx % of Sales ({pct_of_sales*100:.1f}%) exceeds threshold "
                    f"({CapExValidator.MAX_CAPEX_PCT_OF_SALES*100:.0f}%). May include acquisitions."
                )
                confidence *= 0.5
        if previous_capex is not None and previous_capex != 0:
            abs_prev_capex = abs(previous_capex)
            jump_ratio = abs_capex / abs_prev_capex if abs_prev_capex > 0 else float('inf')
            if jump_ratio > CapExValidator.MAX_HISTORICAL_JUMP_RATIO:
                if ppe_change is None or abs(ppe_change) < abs_prev_capex * 0.5:
                    issues.append(
                        f"CapEx jumped {jump_ratio:.1f}x from previous period ({abs_prev_capex:.0f} → {abs_capex:.0f}) "
                        f"without obvious PP&E change justification. May include acquisitions or reflect business changes."
                    )
                    confidence *= 0.7
        if ppe_change is not None and da_value is not None:
            expected_capex = abs(ppe_change) + da_value
            if expected_capex > 0:
                diff_pct = abs(abs_capex - expected_capex) / expected_capex
                if diff_pct > CapExValidator.CAPEX_PPE_TOLERANCE:
                    if diff_pct > 1.0:
                        issues.append(
                            f"CapEx ({abs_capex:.0f}) differs significantly from PP&E change ({ppe_change:.0f}) + D&A ({da_value:.0f}) = {expected_capex:.0f}. "
                            f"Difference: {diff_pct*100:.1f}% (tolerance: {CapExValidator.CAPEX_PPE_TOLERANCE*100:.0f}%). "
                            f"May indicate disposals, impairments, or data quality issues."
                        )
                        confidence *= 0.5
                    else:
                        issues.append(
                            f"CapEx ({abs_capex:.0f}) differs from PP&E change ({ppe_change:.0f}) + D&A ({da_value:.0f}) = {expected_capex:.0f}. "
                            f"Difference: {diff_pct*100:.1f}% (likely due to disposals/impairments, not an error)"
                        )
                        confidence *= 0.85
        return {
            'is_valid': len(issues) == 0,
            'confidence': max(0.0, confidence),
            'issues': issues,
            'suggested_correction': suggested
        }

    @staticmethod
    def validate_da_value(
        da_value: float,
        period_key: str,
        ppe_value: Optional[float] = None,
        revenue: Optional[float] = None,
        previous_da: Optional[float] = None
    ) -> Dict[str, Any]:
        issues = []
        confidence = 1.0
        abs_da = abs(da_value)
        if ppe_value is not None and ppe_value > 0:
            ratio = abs_da / ppe_value
            if ratio > CapExValidator.MAX_DA_TO_PPE_RATIO:
                issues.append(
                    f"D&A ({abs_da:.0f}) is {ratio*100:.1f}% of PP&E ({ppe_value:.0f}), "
                    f"exceeds threshold ({CapExValidator.MAX_DA_TO_PPE_RATIO*100:.0f}%). "
                    f"May indicate unit scale issue or wrong tag."
                )
                confidence *= 0.2
        if revenue is not None and revenue > 0:
            ratio = abs_da / revenue
            if ratio > CapExValidator.MAX_DA_TO_REVENUE_RATIO:
                issues.append(
                    f"D&A ({abs_da:.0f}) is {ratio*100:.1f}% of Revenue ({revenue:.0f}), "
                    f"exceeds threshold ({CapExValidator.MAX_DA_TO_REVENUE_RATIO*100:.0f}%). "
                    f"May indicate unit scale issue."
                )
                confidence *= 0.3
        if previous_da is not None and previous_da != 0:
            abs_prev_da = abs(previous_da)
            jump_ratio = abs_da / abs_prev_da if abs_prev_da > 0 else float('inf')
            if jump_ratio > CapExValidator.MAX_DA_HISTORICAL_JUMP_RATIO:
                issues.append(
                    f"D&A jumped {jump_ratio:.1f}x from previous period ({abs_prev_da:.0f} → {abs_da:.0f}). "
                    f"May indicate unit scale issue or wrong tag."
                )
                confidence *= 0.4
        return {
            'is_valid': len(issues) == 0,
            'confidence': max(0.0, confidence),
            'issues': issues
        }


@dataclass
class CapExValidationContext:
    """Context for validating CapEx values."""
    period_key: str
    ppe_change: Optional[float] = None
    da_value: Optional[float] = None
    revenue: Optional[float] = None
    previous_capex: Optional[float] = None
    ppe_value: Optional[float] = None


@dataclass
class CapExValue:
    """Represents a candidate CapEx value from a specific source."""
    value: float  # Absolute value (always positive)
    source: str  # 'xbrl_tier1', 'xbrl_tier2', 'xbrl_tier3', 'xbrl_components', 'balance_sheet_fallback'
    confidence: float  # 0.0-1.0
    validation_result: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context (tag name, component count, etc.)


class CapExResolver:
    """
    Centralized CapEx resolution using multiple strategies.
    
    This class implements a strategy pattern for resolving CapEx values:
    1. Comprehensive XBRL tags (Tier 1-3)
    2. Component aggregation
    3. Balance sheet fallback
    
    All resolution logic is centralized here following DRY principles.
    """
    
    # Tier definitions for comprehensive tags
    TIER_1_COMPREHENSIVE = {
        'PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets',
        'PaymentsForPropertyPlantAndEquipmentAndIntangibleAssets',
        'PaymentsForAcquisitionOfPropertyPlantAndEquipmentAndIntangibleAssets',
        'InvestmentsInPropertyPlantAndEquipmentAndIntangibleAssets',
    }
    
    TIER_2_PPE_ONLY = {
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'PaymentsForPropertyPlantAndEquipment',
        'PaymentsForPurchaseOfPropertyPlantAndEquipment',
        'InvestmentsInPropertyPlantAndEquipment',
        'CapitalExpendituresForPropertyPlantAndEquipment',
    }
    
    TIER_3_GENERAL = {
        'PaymentsToAcquireProductiveAssets',
        'PaymentsToAcquireOtherProductiveAssets',
        'PaymentsToAcquireOtherPropertyPlantAndEquipment',
        'CapitalExpenditures',
        'CapitalExpendituresNet',
        'PaymentsForCapitalExpenditures',
    }
    
    # Tags to exclude from CapEx
    EXCLUDED_TAGS = {
        # Business acquisitions (not capital expenditures)
        'PaymentsToAcquireBusinessesNetOfCashAcquired',
        'PaymentsToAcquireBusinessesAndIntangibleAssets',
        'PaymentsToAcquireBusinesses',
        'AcquisitionsNetOfCashAcquired',
        # Accrued/liability tags (not actual cash outflow)
        'CapitalExpendituresIncurredButNotYetPaid',
        'CapitalExpendituresDiscontinuedOperations',
    }
    
    def __init__(
        self,
        fact_set: FactSet,
        bs_df: Optional[pd.DataFrame] = None,
        depreciation_series: Optional[pd.Series] = None,
        is_df: Optional[pd.DataFrame] = None,
        ppe_fact_set: Optional[FactSet] = None,
    ):
        """
        Initialize CapEx resolver.
        
        Args:
            fact_set: XBRL fact set containing CapEx facts (typically filter_annual)
            bs_df: Optional balance sheet DataFrame for fallback calculation
            depreciation_series: Optional depreciation Series for fallback calculation
            is_df: Optional income statement DataFrame for validation
            ppe_fact_set: Optional fact set that includes instant facts (e.g. unfiltered)
                for building PP&E Net series. filter_annual() drops instant facts, so
                without this the Net series is empty and fallback uses bs_df (may be Gross).
        """
        self.fact_set = fact_set
        self.bs_df = bs_df
        self.depreciation_series = depreciation_series
        self.is_df = is_df
        self._ppe_fact_set = ppe_fact_set
        
        # Cache for helper methods
        self._ppe_series: Optional[pd.Series] = None
        self._ppe_periods: Optional[List[str]] = None
        self._revenue_series: Optional[pd.Series] = None
        
        # Initialize PP&E: use Net-only from fact set for fallback formula correctness.
        # CapEx = (End Net - Begin Net) + D&A is valid only for NET PP&E; using Gross
        # would overstate CapEx. Prefer ppe_fact_set (includes instant) when provided.
        self._ppe_series, self._ppe_periods = self._build_ppe_net_series()
        if self._ppe_series is None and bs_df is not None and "Property, Plant & Equipment" in bs_df.columns:
            self._ppe_series = bs_df["Property, Plant & Equipment"]
            self._ppe_periods = self._ppe_series.index.tolist()
            log.info(
                "CapExResolver: Using bs_df PP&E column for fallback (no Net series from facts)"
            )
        elif self._ppe_series is not None:
            log.info(
                f"CapExResolver: Built PP&E Net series from facts for {len(self._ppe_periods)} periods"
            )
        
        if is_df is not None and "Revenue" in is_df.columns:
            self._revenue_series = is_df["Revenue"]
    
    def resolve_all_periods(self, xbrl_concepts: List[str]) -> Dict[str, float]:
        """
        Resolve CapEx for all periods using multiple strategies.
        
        This is the main entry point. It:
        1. Extracts XBRL facts
        2. Tries each resolution strategy for each period
        3. Selects the best value when multiple sources are available
        4. Returns final resolved values (negative for cash flow convention)
        
        Args:
            xbrl_concepts: List of XBRL concept names to search for
            
        Returns:
            Dictionary mapping period_key -> CapEx value (negative for cash outflow)
        """
        # Extract and organize XBRL facts
        facts_by_period = self._extract_xbrl_facts(xbrl_concepts)
        
        if not facts_by_period:
            log.warning("CapExResolver: No XBRL facts found for any concepts")
            # Try fallback for all periods if we have balance sheet data
            if self.bs_df is not None and self.depreciation_series is not None:
                return self._resolve_all_periods_fallback()
            return {}
        
        # Resolve each period (process in chronological order to track previous CapEx)
        resolved_values = {}
        all_periods = set(facts_by_period.keys())
        
        # Add periods from depreciation series if available
        if self.depreciation_series is not None:
            all_periods.update(self.depreciation_series.index.astype(str))
        
        # Sort periods chronologically (oldest first) to track previous CapEx
        sorted_periods = sorted(all_periods, key=lambda x: pd.to_datetime(x))
        
        for period_key in sorted_periods:
            candidates = []
            
            # Strategy 1: Comprehensive XBRL tags
            comprehensive_value = self._resolve_from_comprehensive_tags(
                period_key, facts_by_period.get(period_key, {}), resolved_values
            )
            if comprehensive_value is not None:
                candidates.append(comprehensive_value)
            
            # Strategy 2: Component aggregation
            component_value = self._resolve_from_components(
                period_key, facts_by_period.get(period_key, {}), resolved_values
            )
            if component_value is not None:
                candidates.append(component_value)
            
            # Strategy 3: Balance sheet fallback
            fallback_value = self._resolve_from_balance_sheet(period_key, resolved_values)
            if fallback_value is not None:
                candidates.append(fallback_value)
            
            # Select best value
            if candidates:
                best_value = self._select_best_value(candidates, period_key)
                if best_value is not None:
                    # Convert to cash flow convention (negative = cash outflow)
                    resolved_values[period_key] = -abs(best_value.value)
                    self._log_resolution(period_key, best_value, candidates)
            else:
                log.debug(f"CapExResolver: No candidates found for period {period_key}")
        
        return resolved_values
    
    def _extract_xbrl_facts(
        self, xbrl_concepts: List[str]
    ) -> Dict[str, Dict[str, List[Tuple[int, str, Any]]]]:
        """
        Extract and organize XBRL facts by period and tier.
        
        Args:
            xbrl_concepts: List of XBRL concept names
            
        Returns:
            Dictionary mapping period_key -> {
                'comprehensive': [(priority, concept, fact), ...],
                'components': [(priority, concept, fact), ...]
            }
        """
        facts_by_period = defaultdict(lambda: {'comprehensive': [], 'components': []})
        concept_priority = {concept: idx for idx, concept in enumerate(xbrl_concepts)}
        seen_facts = set()
        
        comprehensive_tags = (
            self.TIER_1_COMPREHENSIVE | self.TIER_2_PPE_ONLY | self.TIER_3_GENERAL
        )
        
        for concept in xbrl_concepts:
            # Skip excluded tags
            if self._is_excluded(concept):
                log.debug(f"CapExResolver: Excluding '{concept}' from CapEx")
                continue
            
            facts = self.fact_set.get_by_concept(concept)
            if not facts:
                continue
            
            # Determine if comprehensive or component
            base_concept = self._strip_namespace(concept)
            is_comprehensive = base_concept in comprehensive_tags or concept in comprehensive_tags
            
            for fact in facts:
                # Validate unit
                if not (fact.unit == "USD" or fact.unit.startswith("USD")):
                    continue
                
                # Deduplicate
                fact_key = (concept, str(fact.period.end), fact.value, fact.form or "")
                if fact_key in seen_facts:
                    continue
                seen_facts.add(fact_key)
                
                period_key = str(fact.period.end)
                priority = concept_priority.get(concept, 999)
                form_bonus = 0 if fact.form == "10-K" else 100
                filing_bonus = -(fact.filed.timestamp() if fact.filed else 0)
                
                fact_info = (priority + form_bonus, filing_bonus, concept, fact)
                
                if is_comprehensive:
                    facts_by_period[period_key]['comprehensive'].append(fact_info)
                else:
                    facts_by_period[period_key]['components'].append(fact_info)
        
        # Sort facts by priority
        for period_data in facts_by_period.values():
            period_data['comprehensive'].sort(key=lambda x: (x[0], x[1]))
            period_data['components'].sort(key=lambda x: (x[0], x[1]))
        
        return facts_by_period
    
    def _resolve_from_comprehensive_tags(
        self, period_key: str, period_facts: Dict[str, List[Tuple[int, str, Any]]], resolved_values: Dict[str, float]
    ) -> Optional[CapExValue]:
        """
        Strategy 1: Resolve CapEx from comprehensive XBRL tags (Tier 1-3).
        
        Args:
            period_key: Period to resolve
            period_facts: Facts for this period
            
        Returns:
            CapExValue if found, None otherwise
        """
        comprehensive_facts = period_facts.get('comprehensive', [])
        if not comprehensive_facts:
            return None
        
        # Group by tier
        facts_by_tier = defaultdict(list)
        for fact_info in comprehensive_facts:
            _, _, concept, _ = fact_info
            tier = self._get_tag_tier(concept)
            facts_by_tier[tier].append(fact_info)
        
        # Try each tier in order
        for tier in sorted(facts_by_tier.keys()):
            tier_facts = facts_by_tier[tier]
            
            for fact_info in tier_facts:
                _, _, concept, fact = fact_info
                candidate_value = fact.value
                
                # Skip excluded tags
                if self._is_excluded(concept):
                    continue
                
                # Get validation context
                context = self._get_validation_context(period_key, resolved_values)
                
                # Validate (advisory only: used for logging and confidence, never to reject XBRL)
                validation_result = CapExValidator.validate_capex_value(
                    candidate_value,
                    period_key,
                    context.ppe_change,
                    context.da_value,
                    context.revenue,
                    context.previous_capex,
                    context.ppe_value,
                )
                
                # Edgartools-aligned: use reported XBRL when present; do not reject on validation.
                # Only skip clearly invalid facts (non-numeric or unconvertible).
                try:
                    numeric_value = float(candidate_value) if candidate_value is not None else None
                except (ValueError, TypeError):
                    log.debug(
                        f"CapExResolver: Skipping Tier {tier} tag '{concept}' for {period_key} "
                        f"(non-numeric value)"
                    )
                    continue
                if numeric_value is None:
                    continue
                
                # Found valid tag - use it (fallback only when no XBRL)
                return CapExValue(
                    value=abs(numeric_value),
                    source=f'xbrl_tier{tier}',
                    confidence=validation_result['confidence'],
                    validation_result=validation_result,
                    metadata={'concept': concept, 'tier': tier},
                )
        
        return None
    
    def _resolve_from_components(
        self, period_key: str, period_facts: Dict[str, List[Tuple[int, str, Any]]], resolved_values: Dict[str, float]
    ) -> Optional[CapExValue]:
        """
        Strategy 2: Resolve CapEx by aggregating component tags.
        
        Args:
            period_key: Period to resolve
            period_facts: Facts for this period
            
        Returns:
            CapExValue if found, None otherwise
        """
        component_facts = period_facts.get('components', [])
        if not component_facts:
            return None
        
        total_value = 0.0
        used_concepts = []
        seen_component_facts = set()
        component_details = []
        
        for _, _, concept, fact in component_facts:
            # Skip excluded tags
            if self._is_excluded(concept):
                continue
            
            # Deduplicate
            fact_signature = (concept, fact.value)
            if fact_signature in seen_component_facts:
                continue
            seen_component_facts.add(fact_signature)
            
            try:
                fact_value = float(fact.value) if fact.value is not None else 0.0
                total_value += fact_value
                used_concepts.append(concept)
                component_details.append({
                    'concept': concept,
                    'value': fact_value,
                })
            except (ValueError, TypeError):
                continue
        
        if not used_concepts:
            return None
        
        # Validate aggregated value (advisory only: for logging and confidence, never to reject)
        context = self._get_validation_context(period_key, resolved_values)
        validation_result = CapExValidator.validate_capex_value(
            total_value,
            period_key,
            context.ppe_change,
            context.da_value,
            context.revenue,
            context.previous_capex,
            context.ppe_value,
        )
        
        # Edgartools-aligned: always use aggregated components when present; fallback only when no XBRL
        return CapExValue(
            value=abs(total_value),
            source='xbrl_components',
            confidence=validation_result['confidence'],
            validation_result=validation_result,
            metadata={
                'component_count': len(used_concepts),
                'components': used_concepts[:5],  # First 5 for logging
            },
        )
    
    def _resolve_from_balance_sheet(self, period_key: str, resolved_values: Dict[str, float]) -> Optional[CapExValue]:
        """
        Strategy 3: Resolve CapEx using balance sheet fallback.
        
        Formula: CapEx = (Ending PP&E - Beginning PP&E) + D&A
        
        Includes detection of acquisitions in PP&E_change.
        
        Args:
            period_key: Period to resolve
            
        Returns:
            CapExValue if calculated, None otherwise
        """
        if self._ppe_series is None or self.depreciation_series is None:
            return None
        
        # Get PP&E values
        ending_ppe = self._get_aligned_value(self._ppe_series, period_key, self._ppe_periods)
        if ending_ppe is None:
            return None
        
        # Find beginning PP&E
        period_dt = pd.to_datetime(period_key)
        beginning_ppe = None
        beginning_period = None
        
        for ppe_period in sorted(self._ppe_periods, reverse=True):
            try:
                ppe_period_dt = pd.to_datetime(ppe_period)
                if ppe_period_dt < period_dt:
                    diff_days = abs((period_dt - ppe_period_dt).days)
                    if 300 <= diff_days <= 400:  # ~1 year
                        beginning_ppe = self._ppe_series.get(ppe_period)
                        beginning_period = ppe_period
                        if beginning_ppe is not None and not pd.isna(beginning_ppe):
                            break
            except (ValueError, TypeError):
                continue
        
        if beginning_ppe is None or pd.isna(beginning_ppe):
            return None
        
        # Get D&A
        da_value = self._get_aligned_value(
            self.depreciation_series, period_key, self.depreciation_series.index.tolist()
        )
        if da_value is None:
            return None
        
        try:
            ending_ppe_float = float(ending_ppe)
            beginning_ppe_float = float(beginning_ppe)
            da_float = float(da_value)
            
            # Calculate PP&E change
            ppe_change = ending_ppe_float - beginning_ppe_float
            
            # Validate D&A before using
            ppe_value = ending_ppe_float
            revenue = self._get_revenue_for_period(period_key)
            prev_da = self._get_previous_da(period_key)
            
            da_validation = CapExValidator.validate_da_value(
                da_float, period_key, ppe_value, revenue, prev_da
            )
            
            # Reject if D&A has severe issues
            if da_validation['confidence'] < 0.2:
                log.debug(
                    f"CapExResolver: Rejecting fallback for {period_key} due to low D&A confidence"
                )
                return None
            
            # Calculate CapEx
            capex = ppe_change + da_float
            
            # Detect acquisitions in PP&E_change
            acquisition_confidence_adjustment = self._detect_acquisitions_in_ppe_change(
                period_key, ppe_change, da_float
            )
            
            # Validate calculated CapEx
            context = self._get_validation_context(period_key, resolved_values)
            validation_result = CapExValidator.validate_capex_value(
                capex,
                period_key,
                ppe_change,
                da_float,
                context.revenue,
                context.previous_capex,
                ppe_value,
            )
            
            # Adjust confidence based on acquisition detection
            confidence = validation_result['confidence'] * acquisition_confidence_adjustment
            
            # Reject if both D&A and CapEx have severe issues
            if not da_validation['is_valid'] and validation_result['confidence'] < 0.3:
                if da_validation['confidence'] < 0.4:
                    log.debug(
                        f"CapExResolver: Rejecting fallback for {period_key} "
                        f"(both D&A and CapEx have severe issues)"
                    )
                    return None
            
            return CapExValue(
                value=abs(capex),
                source='balance_sheet_fallback',
                confidence=confidence,
                validation_result=validation_result,
                metadata={
                    'ppe_change': ppe_change,
                    'da_value': da_float,
                    'beginning_period': beginning_period,
                    'acquisition_detected': acquisition_confidence_adjustment < 0.8,
                },
            )
        except (ValueError, TypeError) as e:
            log.debug(f"CapExResolver: Error calculating fallback for {period_key}: {e}")
            return None
    
    def _select_best_value(
        self, candidates: List[CapExValue], period_key: str
    ) -> Optional[CapExValue]:
        """
        Select the best CapEx value from multiple candidates.
        
        Priority:
        1. XBRL values over fallback
        2. Higher confidence
        3. Higher tier (Tier 1 > Tier 2 > Tier 3 > components)
        
        Args:
            candidates: List of CapExValue candidates
            period_key: Period key for logging
            
        Returns:
            Best CapExValue or None
        """
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Separate XBRL from fallback
        xbrl_candidates = [c for c in candidates if c.source != 'balance_sheet_fallback']
        fallback_candidates = [c for c in candidates if c.source == 'balance_sheet_fallback']
        
        # Prefer XBRL over fallback (edgartools-aligned: use reported XBRL when present)
        if xbrl_candidates:
            candidates_to_consider = xbrl_candidates
        else:
            candidates_to_consider = fallback_candidates
        
        # Sort by: tier priority, then confidence
        def sort_key(candidate: CapExValue) -> Tuple[int, float]:
            tier_priority = {
                'xbrl_tier1': 1,
                'xbrl_tier2': 2,
                'xbrl_tier3': 3,
                'xbrl_components': 4,
                'balance_sheet_fallback': 5,
            }.get(candidate.source, 999)
            return (tier_priority, -candidate.confidence)  # Negative for descending
        
        candidates_to_consider.sort(key=sort_key)
        best = candidates_to_consider[0]
        
        # Log selection: used XBRL vs used fallback (edgartools-aligned auditability)
        if len(candidates) > 1:
            log.info(
                f"CapExResolver: Selected {best.source} for {period_key} "
                f"(confidence: {best.confidence:.2f}) from {len(candidates)} candidates"
            )
        else:
            log.debug(
                f"CapExResolver: Used {best.source} for {period_key} (confidence: {best.confidence:.2f})"
            )
        
        return best
    
    def _get_validation_context(self, period_key: str, resolved_values: Dict[str, float]) -> CapExValidationContext:
        """Build validation context for a period."""
        ppe_change = self._get_ppe_change_for_period(period_key)
        da_value = self._get_da_for_period(period_key)
        revenue = self._get_revenue_for_period(period_key)
        ppe_value = self._get_ppe_for_period(period_key)
        
        # Get previous CapEx from resolved values
        previous_capex = None
        try:
            period_dt = pd.to_datetime(period_key)
            previous_periods = []
            
            for k in resolved_values.keys():
                try:
                    k_dt = pd.to_datetime(k)
                    if k_dt < period_dt:
                        previous_periods.append((k, k_dt))
                except (ValueError, TypeError):
                    continue
            
            if previous_periods:
                previous_periods.sort(key=lambda x: x[1], reverse=True)
                prev_key = previous_periods[0][0]
                prev_value = resolved_values.get(prev_key)
                if prev_value is not None:
                    previous_capex = abs(prev_value)  # Use absolute value for comparison
        except (ValueError, TypeError):
            pass
        
        return CapExValidationContext(
            period_key=period_key,
            ppe_change=ppe_change,
            da_value=da_value,
            revenue=revenue,
            previous_capex=previous_capex,
            ppe_value=ppe_value,
        )
    
    def _detect_acquisitions_in_ppe_change(
        self, period_key: str, ppe_change: float, da_value: float
    ) -> float:
        """
        Detect if PP&E_change includes acquisitions.
        
        Returns confidence adjustment factor (0.0-1.0).
        Lower value indicates acquisitions likely included.
        """
        # If CapEx would be >3x typical CapEx/D&A ratio, likely includes acquisitions
        if da_value > 0:
            capex_to_da = abs(ppe_change) / da_value
            if capex_to_da > 3.0:
                log.warning(
                    f"CapExResolver: PP&E_change for {period_key} likely includes acquisitions "
                    f"(PP&E_change/D&A ratio: {capex_to_da:.1f}x)"
                )
                return 0.5  # Reduce confidence significantly
        
        # Check if PP&E_change is unusually large relative to historical patterns
        # (Would need historical data - for now, use simple heuristic)
        if abs(ppe_change) > da_value * 5.0:
            log.warning(
                f"CapExResolver: PP&E_change for {period_key} is unusually large "
                f"(PP&E_change: {ppe_change:.0f}, D&A: {da_value:.0f})"
            )
            return 0.7  # Reduce confidence moderately
        
        return 1.0  # No acquisitions detected
    
    def _resolve_all_periods_fallback(self) -> Dict[str, float]:
        """Resolve all periods using only fallback method."""
        resolved_values = {}
        
        if self.depreciation_series is None:
            return resolved_values
        
        for period_key in self.depreciation_series.index.astype(str):
            fallback_value = self._resolve_from_balance_sheet(period_key, resolved_values)
            if fallback_value is not None:
                resolved_values[period_key] = -abs(fallback_value.value)
        
        return resolved_values
    
    # Helper methods (similar to CashFlowStatement)
    
    def _get_ppe_change_for_period(self, period_key: str) -> Optional[float]:
        """Get PP&E change for a period."""
        if self._ppe_series is None:
            return None
        
        ending_ppe = self._get_aligned_value(self._ppe_series, period_key, self._ppe_periods)
        if ending_ppe is None:
            return None
        
        period_dt = pd.to_datetime(period_key)
        beginning_ppe = None
        
        for prev_period in sorted(self._ppe_periods, reverse=True):
            try:
                prev_dt = pd.to_datetime(prev_period)
                if prev_dt < period_dt:
                    diff_days = abs((period_dt - prev_dt).days)
                    if 300 <= diff_days <= 400:
                        beginning_ppe = self._ppe_series.get(prev_period)
                        if beginning_ppe is not None and not pd.isna(beginning_ppe):
                            break
            except (ValueError, TypeError):
                continue
        
        if beginning_ppe is None:
            return None
        
        try:
            return float(ending_ppe) - float(beginning_ppe)
        except (ValueError, TypeError):
            return None
    
    def _get_ppe_for_period(self, period_key: str) -> Optional[float]:
        """Get PP&E value for a period."""
        if self._ppe_series is None:
            return None
        
        ppe_value = self._get_aligned_value(self._ppe_series, period_key, self._ppe_periods)
        return float(ppe_value) if ppe_value is not None else None
    
    def _get_revenue_for_period(self, period_key: str) -> Optional[float]:
        """Get Revenue for a period."""
        if self._revenue_series is None:
            return None
        
        revenue_value = self._get_aligned_value(
            self._revenue_series, period_key, self._revenue_series.index.tolist()
        )
        return float(revenue_value) if revenue_value is not None else None
    
    def _get_da_for_period(self, period_key: str) -> Optional[float]:
        """Get D&A value for a period."""
        if self.depreciation_series is None:
            return None
        
        da_value = self._get_aligned_value(
            self.depreciation_series, period_key, self.depreciation_series.index.tolist()
        )
        return float(da_value) if da_value is not None else None
    
    def _get_previous_da(self, period_key: str) -> Optional[float]:
        """Get previous period D&A."""
        if self.depreciation_series is None:
            return None
        
        try:
            period_dt = pd.to_datetime(period_key)
            previous_periods = []
            
            for k in self.depreciation_series.index:
                try:
                    k_dt = pd.to_datetime(k)
                    if k_dt < period_dt:
                        previous_periods.append((k, k_dt))
                except (ValueError, TypeError):
                    continue
            
            if not previous_periods:
                return None
            
            previous_periods.sort(key=lambda x: x[1], reverse=True)
            prev_key = previous_periods[0][0]
            prev_value = self.depreciation_series.get(prev_key)
            return float(prev_value) if prev_value is not None else None
        except (ValueError, TypeError):
            return None
    
    def _build_ppe_net_series(
        self,
    ) -> Tuple[Optional[pd.Series], Optional[List[str]]]:
        """
        Build PP&E series from PropertyPlantAndEquipmentNet only (no Gross).
        
        The fallback formula CapEx = (End PP&E - Begin PP&E) + D&A is correct only
        when PP&E is NET (after accumulated depreciation). Using Gross PP&E would
        overstate CapEx. This method ensures we use only Net for the fallback.
        
        Returns:
            (Series indexed by period_key, list of period keys) or (None, None) if empty.
        """
        # Use ppe_fact_set when provided (includes instant facts); filter_annual drops them
        fact_source = self._ppe_fact_set if self._ppe_fact_set is not None else self.fact_set
        if fact_source is None:
            return None, None
        
        net_concepts = ["PropertyPlantAndEquipmentNet", "us-gaap_PropertyPlantAndEquipmentNet"]
        period_candidates: Dict[str, List[Tuple[int, float, Optional[datetime]]]] = defaultdict(list)
        total_net_facts = 0
        
        for concept in net_concepts:
            facts = fact_source.get_by_concept(concept)
            if not facts:
                continue
            total_net_facts += len(facts)
            for fact in facts:
                if not (fact.unit == "USD" or (getattr(fact.unit, "startswith", None) and fact.unit.startswith("USD"))):
                    continue
                try:
                    period_key = str(fact.period.end)
                    value = float(fact.value)
                except (ValueError, TypeError):
                    continue
                form_priority = 0 if fact.form == "10-K" else 1
                filed_ts = fact.filed.timestamp() if getattr(fact, "filed", None) else 0.0
                period_candidates[period_key].append((form_priority, value, getattr(fact, "filed", None)))
        
        if not period_candidates:
            if self._ppe_fact_set is not None and total_net_facts == 0:
                log.info(
                    "CapExResolver: ppe_fact_set has no PropertyPlantAndEquipmentNet facts; "
                    "company may report only Gross PP&E; using bs_df fallback when available"
                )
            return None, None
        
        # Best value per period: prefer 10-K, then most recent filed
        best_by_period: Dict[str, float] = {}
        for period_key, candidates in period_candidates.items():
            candidates.sort(
                key=lambda x: (x[0], -(x[2].timestamp() if x[2] is not None else 0.0))
            )
            best_by_period[period_key] = candidates[0][1]
        
        series = pd.Series(best_by_period)
        periods = series.index.tolist()
        return series, periods
    
    def _get_aligned_value(
        self, series: pd.Series, period_key: str, available_periods: List[str]
    ) -> Optional[Any]:
        """Get value from series aligned to period_key."""
        # Try exact match first
        if period_key in series.index:
            return series[period_key]
        
        # Try string conversion
        try:
            period_dt = pd.to_datetime(period_key)
            for period in available_periods:
                try:
                    period_dt_candidate = pd.to_datetime(period)
                    if abs((period_dt - period_dt_candidate).days) < 5:
                        return series.get(period)
                except (ValueError, TypeError):
                    continue
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _is_excluded(self, concept: str) -> bool:
        """Check if concept should be excluded."""
        base_name = self._strip_namespace(concept)
        return base_name in self.EXCLUDED_TAGS or concept in self.EXCLUDED_TAGS
    
    def _strip_namespace(self, concept: str) -> str:
        """Strip namespace prefix from concept."""
        base_name = concept
        for prefix in ['us-gaap_', 'us-gaap:', 'dei:', 'dei_']:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break
        return base_name
    
    def _get_tag_tier(self, concept: str) -> int:
        """Get priority tier for a concept."""
        base_concept = self._strip_namespace(concept)
        
        if base_concept in self.TIER_1_COMPREHENSIVE:
            return 1
        elif base_concept in self.TIER_2_PPE_ONLY:
            return 2
        elif base_concept in self.TIER_3_GENERAL:
            return 3
        else:
            return 999
    
    def _log_resolution(
        self, period_key: str, selected: CapExValue, all_candidates: List[CapExValue]
    ):
        """Log resolution decision."""
        log.info(
            f"CapExResolver: Resolved {period_key} -> {selected.source} "
            f"(value: {selected.value:.0f}, confidence: {selected.confidence:.2f})"
        )
        
        if len(all_candidates) > 1:
            log.debug(
                f"CapExResolver: Other candidates for {period_key}: "
                + ", ".join([
                    f"{c.source}({c.value:.0f}, conf={c.confidence:.2f})"
                    for c in all_candidates if c != selected
                ])
            )
